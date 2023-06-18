import json
import logging
import os
from collections import OrderedDict

from transformers import (
    AdapterTrainer,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)
from transformers.adapters.configuration import (
    PfeifferConfig,
    AdapterConfig,
    CongaterV2Config,
)
from transformers.adapters.training import setup_adapter_training

from torchinfo import summary

from arguments import get_args
from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
from tasks.superglue.dataset import SuperGlueDataset
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from training.utils import map_omega_grid

logger = logging.getLogger(__name__)


def get_trainer(args):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        congater_args,
    ) = get_args()
    if congater_args.debug_congater:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        fusion_args.fusion_load_dir = "../" + fusion_args.fusion_load_dir
        # out dir
        # training_args.output_dir = "../" + training_args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.task_name.lower() in GLUE_DATASETS:
        dataset = GlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() in SUPERGLUE_DATASETS:
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression and not dataset.multiple_choice:
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
        model = get_model(
            args=args, task_type=TaskType.SEQUENCE_CLASSIFICATION, config=config
        )
    else:
        model = get_model(args=args, task_type=TaskType.MULTIPLE_CHOICE, config=config)

    adapter_setup = None
    if fusion_args.train_fusion:
        if adapter_args.train_adapter:
            raise ValueError(
                "Fusion training is currently not supported in adapter training mode."
                "Set --train_adapter to False to enable fusion training"
            )
        af_config = json.load(open(fusion_args.fusion_load_dir))

        if fusion_args.fusion_type != "dynamic":
            # as OrderedDict to preserve order.
            # then, set data_args.task_name as first task
            # used in layer.py - last adapter is needed as task adapter
            af_config = OrderedDict(af_config)
            af_config.move_to_end(data_args.task_name, last=True)

        if fusion_args.fusion_adapter_config == "pfeiffer":
            adapter_config = PfeifferConfig()
        elif fusion_args.fusion_adapter_config == "congaterV2":
            adapter_config = CongaterV2Config()
        else:
            raise ValueError(
                "Only pfeiffer & CongaterV2 is currently supported for fusion training."
                "Set --fusion_adapter_config to pfeiffer"
            )

        for t, task in af_config.items():
            if t == "SELF":
                print("Changing SELF to task name")
                # replace SELF with task name
                af_config[data_args.task_name] = af_config["SELF"].replace(
                    "SELF", data_args.task_name
                )
                del af_config["SELF"]
                break

        for _, task in af_config.items():
            task = task.split("/")[-3]
            seed = af_config[task][-1]
            logger.info(task)
            # use max_train_pct for task, else 100
            if task == data_args.task_name:
                af_config[task] = (
                    af_config[task][:-1] + str(data_args.max_train_pct) + "/" + seed
                )
            else:
                af_config[task] = af_config[task][:-1] + "100" + "/" + seed

        logger.info(af_config)
        for _, adapter_dir in af_config.items():
            logger.info(adapter_dir)
            model.load_adapter(
                f"{os.path.expanduser('~')}/congater-fusion/src/" + adapter_dir,
                config=adapter_config,
                with_head=fusion_args.fusion_with_head,
            )
        adapter_setup = [list(af_config.keys())]

        # Add a fusion layer and tell the model to train fusion
        if congater_args.congosition_type:
            model.add_congosition_v1(adapter_setup[0], fusion_args.fusion_type)
            model.train_adapter_fusion(
                adapter_setup, unfreeze_adapters=fusion_args.fusion_unfreeze_adapters
            )
            # if data_args.task_name == "wsc":
                # if dataset.multiple_choice:
                #     model.add_multiple_choice_head(data_args.task_name, num_choices=2)
                # else:
                #     model.add_classification_head(
                #         data_args.task_name,
                #         num_labels=dataset.num_labels,
                #         id2label={i: v for i, v in enumerate(dataset.label_list)}
                #         if not dataset.is_regression
                #         else None,
                #     )
        else:
            model.add_adapter_fusion(adapter_setup[0], fusion_args.fusion_type)
            model.train_adapter_fusion(
                adapter_setup, unfreeze_adapters=fusion_args.fusion_unfreeze_adapters
            )

    elif adapter_args.train_adapter:
        if data_args.omega_grid and congater_args.congosition_type == "omega_grid":
            omega_grid = map_omega_grid(
                config=data_args.omega_grid, 
                seed=training_args.seed,
                adapter_type=adapter_args.adapter_config,
            )
            for adapter_dir, _ in omega_grid.items():
                logger.info(adapter_dir)
                model.load_adapter(
                    f"{os.path.expanduser('~')}/congater-fusion/src/" + adapter_dir,
                    with_head=False,
                )
            # get tasks from omega_grid
            source_tasks = [l.split("/")[2] for l in list(omega_grid.keys())]
            # create dict: task -> omega
            omega_grid = {l.split("/")[2]:i for l, i in list(omega_grid.items())}
            model.add_congosition_v1(source_tasks, congater_args.congosition_type, grid_values=omega_grid)
            model.train_adapter_fusion(
                [source_tasks], unfreeze_adapters=False
            )
        if data_args.eval_adapter:
            config = AdapterConfig.load(
                training_args.output_dir + "/adapter_config.json"
            )
            # config = AdapterConfig.load("pfeiffer")
            config.omega = model_args.omega
            if data_args.train_probing_head:
                model.load_adapter(
                    training_args.output_dir,
                    config=config,
                    with_head=True,
                )
                model.train()
                # model.freeze_model(True)
                if data_args.source_task:
                    model.train_adapter([data_args.source_task])
                    model.set_active_adapters(data_args.source_task)
                else:
                    model.train_adapter([data_args.task_name])
                    model.set_active_adapters(data_args.task_name)
                model.freeze_model(True)
                if data_args.source_task:
                    head_name = data_args.source_task
                else:
                    head_name = data_args.task_name
                if dataset.multiple_choice:
                        model.add_multiple_choice_head(head_name, num_choices=2, overwrite_ok=True)
                else:
                    model.add_classification_head(
                        head_name,
                        num_labels=dataset.num_labels,
                        overwrite_ok=True,
                        id2label={i: v for i, v in enumerate(dataset.label_list)}
                        if not dataset.is_regression
                        else None,
                    )
            else:
                model.load_adapter(
                    training_args.output_dir,
                    config=config,
                    with_head=True,
                )
                # model.load_adapter("sentiment/sst-2@ukp", config=config)
                model.train_adapter([data_args.task_name])
                model.set_active_adapters(data_args.task_name)
        else:
            if dataset.multiple_choice:
                model.add_multiple_choice_head(data_args.task_name, num_choices=2)
            else:
                model.add_classification_head(
                    data_args.task_name,
                    num_labels=dataset.num_labels,
                    id2label={i: v for i, v in enumerate(dataset.label_list)}
                    if not dataset.is_regression
                    else None,
                )
            # Setup adapters
            # TODO: setup variable omega for training also (not only eval mode, i.e. data_args.eval_adapter = True)
            if not data_args.omega_grid:
                setup_adapter_training(model, adapter_args, data_args.task_name)
    else:
        if adapter_args.load_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")
            # print(n, len(p))

    trainer_cls = (
        AdapterTrainer
        if (adapter_args.train_adapter or fusion_args.train_fusion)
        # and not data_args.train_probing_head
        else Trainer
    )

    # early stopping
    if model_args.early_stopping:
        logger.info(
            "Early stopping is enabled with patience %d",
            model_args.early_stopping_patience,
        )
        early_stopping_callback = [EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience
        )]
    else:
        early_stopping_callback = []

    # wandb callback
    from transformers.integrations import WandbCallback

    # os.environ["WANDB_WATCH"] = "all"
    # os.environ["WANDB_LOG_MODEL "] = "true"
    # wandb_callback = WandbCallback()

    logger.info(summary(model, depth=2))

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks=early_stopping_callback,
    )

    return trainer, model, dataset, adapter_setup
