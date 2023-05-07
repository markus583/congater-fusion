import json
import logging
import os

from transformers import (
    AdapterTrainer,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)
from transformers.adapters.configuration import PfeifferConfig, AdapterConfig
from transformers.adapters.training import setup_adapter_training

from arguments import get_args
from model.utils import TaskType, get_model
from tasks.superglue.dataset import SuperGlueDataset
from tasks.superglue.dataset_record import SuperGlueDatasetForRecord

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, adapter_args, fusion_args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.task_name == "recordTEST":
        dataset = SuperGlueDatasetForRecord(tokenizer, data_args, training_args)
    else:
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    if not dataset.multiple_choice:
        model = get_model(args=args, task_type=TaskType.SEQUENCE_CLASSIFICATION, config=config)
    else:
        # TODO: check fix_bert from original 
        model = get_model(args=args, task_type=TaskType.MULTIPLE_CHOICE, config=config)

    adapter_setup = None
    if fusion_args.train_fusion:
        if adapter_args.train_adapter:
            raise ValueError(
                "Fusion training is currently not supported in adapter training mode."
                "Set --train_adapter to False to enable fusion training"
            )
        af_config = json.load(open(fusion_args.fusion_load_dir))
        if data_args.max_train_pct != 100:
            seed = af_config[data_args.task_name][-1]
            af_config[data_args.task_name] = (
                af_config[data_args.task_name][:-1]
                + str(data_args.max_train_pct)
                + "/"
                + seed
            )
        if fusion_args.fusion_adapter_config == "pfeiffer":
            adapter_config = PfeifferConfig()
        else:
            # TODO: implement for ConGater
            raise ValueError(
                "Only pfeiffer is currently supported for fusion training."
                "Set --fusion_adapter_config to pfeiffer"
            )
        for _, adapter_dir in af_config.items():
            # TODO: switch to superglue/combination
            model.load_adapter(
                f"{os.path.expanduser('~')}/congater-fusion/adapter-reproduction/glue/"
                + adapter_dir,
                config=adapter_config,
                with_head=fusion_args.fusion_with_head,
            )
        adapter_setup = [list(af_config.keys())]

        # Add a fusion layer and tell the model to train fusion
        model.add_adapter_fusion(adapter_setup[0], fusion_args.fusion_type)
        model.train_adapter_fusion(
            adapter_setup, unfreeze_adapters=fusion_args.fusion_unfreeze_adapters
        )

    elif adapter_args.train_adapter:
        if data_args.eval_adapter:
            config = AdapterConfig.load(training_args.output_dir + "/adapter_config.json")
            # config = AdapterConfig.load("pfeiffer")
            config.omega = model_args.omega
            model.load_adapter(
               training_args.output_dir,
               config=config,
               with_head=True,
            )
            model.train_adapter([data_args.task_name])
            model.set_active_adapters(data_args.task_name)
        else:
            if dataset.multiple_choice:
                model.add_multiple_choice_head(
                    data_args.task_name or "superglue",
                    num_choices=2
                )
            else:
                model.add_classification_head(
                    data_args.task_name or "superglue",
                    num_labels=dataset.num_labels,
                    id2label={i: v for i, v in enumerate(dataset.label_list)}
                    if not dataset.is_regression
                    else None,
                )
            # Setup adapters
            # TODO: setup variable omega for training also (not only eval mode, i.e. data_args.eval_adapter = True)
            setup_adapter_training(model, adapter_args, data_args.task_name or "superglue")
    else:
        if adapter_args.load_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )
    # TODO: fix or remove
    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")
            # print(n)

    trainer_cls = (
        AdapterTrainer
        if adapter_args.train_adapter or fusion_args.train_fusion
        else Trainer
    )

    # early stopping
    if model_args.early_stopping:
        logger.info(
            "Early stopping is enabled with patience %d",
            model_args.early_stopping_patience,
        )
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience
        )
    else:
        early_stopping_callback = None

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks=[early_stopping_callback],
    )

    return trainer, model, dataset, adapter_setup
