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
    AutoModelForSequenceClassification,
)
from transformers.adapters.configuration import (
    PfeifferConfig,
    AdapterConfig,
    CongaterV2Config,
)
from transformers.adapters.training import setup_adapter_training
from transformers.adapters import AutoAdapterModel
import datasets

from torchinfo import summary

from arguments_multi import get_args
from model.utils import TaskType, AUTO_MODELS, get_model
from model.model_multi import MultitaskModel
from model.model_adapter_multi import MultitaskAdapterModel
from tasks.glue.dataset import GlueDataset
from tasks.abstract_task import TaskDataset, build_compute_metrics_fn
from tasks.superglue.dataset import SuperGlueDataset
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from training.utils import map_omega_grid
from training.multi_trainer import MultiTrainer

logger = logging.getLogger(__name__)


class AutoTask:
    @classmethod
    def get(self, task_name: int):
        if task_name in GLUE_DATASETS or task_name in SUPERGLUE_DATASETS:
            return TaskDataset(task_name)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )


def get_trainer(args):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        congater_args,
    ) = get_args()
    if len(data_args.tasks) == 1:
        # convert ["['rte', 'mrpc', ...]"] to ['rte', 'mrpc', ...]
        data_args.tasks = eval(data_args.tasks[0])
    if len(data_args.eval_tasks) == 1:
        data_args.eval_tasks = eval(data_args.eval_tasks[0])
    if data_args.train_tasks is None or len(data_args.train_tasks) == 0:
        data_args.train_tasks = data_args.tasks

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

    # Gets the training/test/validation datasets.
    train_datasets_cls = [AutoTask.get(task) for task in data_args.train_tasks]
    train_datasets = [
        ds.get_dataset(
            split="train",
            n_obs=data_args.n_train,
        )
        for ds in train_datasets_cls
    ]
    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = {
        task: AutoTask.get(task).get_dataset(
            split="validation",
            n_obs=data_args.n_val,  ## for cross lingual transfer, some task only have test set.
        )
        for task in data_args.eval_tasks
    }

    test_datasets = {
        task: AutoTask.get(task).get_dataset(
            split="test",
            n_obs=data_args.n_test,  ## for cross lingual transfer, some task only have test set.
        )
        for task in data_args.eval_tasks
    }
    compute_metrics_fn = build_compute_metrics_fn(data_args.eval_tasks)

    model_config_dict = {
        task.name: AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=task.num_labels,
            finetuning_task=task.name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        for task in train_datasets_cls
    }
    if adapter_args.train_adapter:
        model_type_dict = {
            task.name: AutoAdapterModel
            for task in train_datasets_cls
        }
        model = MultitaskAdapterModel.create(
            model_name=model_args.model_name_or_path,
            model_type_dict=model_type_dict,
            model_config_dict=model_config_dict,
            dataset_cls=train_datasets_cls,
        )
    else:
        model_type_dict = {
            task.name: AUTO_MODELS[TaskType.SEQUENCE_CLASSIFICATION]
            if not task.multiple_choice
            else AUTO_MODELS[TaskType.MULTIPLE_CHOICE]
            for task in train_datasets_cls
        }
        model = MultitaskModel.create(
            model_name=model_args.model_name_or_path,
            model_type_dict=model_type_dict,
            model_config_dict=model_config_dict,
        )
    model.print_parameter_info()

    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")
            # print(n, len(p))

    # trainer_cls = (
    #     AdapterTrainer
    #     if (adapter_args.train_adapter or fusion_args.train_fusion)
    #     # and not data_args.train_probing_head
    #     else Trainer
    # )

    # # early stopping
    # if model_args.early_stopping:
    #     logger.info(
    #         "Early stopping is enabled with patience %d",
    #         model_args.early_stopping_patience,
    #     )
    #     early_stopping_callback = [
    #         EarlyStoppingCallback(
    #             early_stopping_patience=model_args.early_stopping_patience
    #         )
    #     ]
    # else:
    #     early_stopping_callback = []

    logger.info(summary(model, depth=10))

        trainer = MultiTrainer(
            model=model,
            config=model_config_dict,  # TODO: change to model.config pattern
            args=training_args,
            train_dataset=train_dataset
            eval_dataset=eval_datasets,
            data_collator=TaskCollator(
                tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores
            ),
            compute_metrics=None,
            multi_task_compute_metrics=compute_metrics_fn,
            data_args=data_args,
            dataset_sizes=dataset_sizes,
            adapter_config=adapter_config,
        )

    return trainer, model, dataset, adapter_setup
