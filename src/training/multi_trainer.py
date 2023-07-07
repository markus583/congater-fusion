"""Implements a T5 trainer class doing training and evaluation."""

import collections
import math

import numpy as np
import os
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, logging
from transformers import Trainer
from transformers.file_utils import is_torch_tpu_available
from transformers.integrations import hp_params
import wandb
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput
from transformers.trainer_utils import set_seed

_use_apex = False
_use_native_amp = False
# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset

from utils import use_task_specific_params, reset_config
from ..tasks import MultiTaskBatchSampler

logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class MultiTrainer(Trainer):
    def __init__(
        self,
        config=None,
        data_args=None,
        dataset_sizes=None,
        adapter_config=None,
        multi_task_compute_metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        self.vocab_size = self.config.vocab_size

        if self.args.label_smoothing != 0 or (
            self.data_args is not None and self.data_args.ignore_pad_token_for_loss
        ):
            assert (
                self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warn(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.config.pad_token_id
        )
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        multitask_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_sampler=multitask_sampler,
            collate_fn=self.data_collator,
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available() and xm.xrt_world_size() > 1:
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        elif self.args.local_rank != -1:
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0
        return MultiTaskBatchSampler(
            self.dataset_sizes,
            self.args.train_batch_size,
            self.args.temperature,
            rank=rank,
            num_replicas=num_replicas,
        )
        
    def evaluate(
        self,
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        is_test=False,
        ignore_keys=None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        results = {}
        if eval_datasets is None:
            eval_datasets = self.eval_dataset

        for eval_task, eval_dataset in eval_datasets.items():
            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            model_config = self.model.config

            use_task_specific_params(self.model, eval_task)

            if eval_dataset is not None and not isinstance(
                eval_dataset, collections.abc.Sized
            ):
                raise ValueError("eval_dataset must implement __len__")

            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
            )
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            if is_test:
                tasks_metric = {
                    "test/" + eval_task + "_" + k: v for k, v in output.metrics.items()
                }
            else:
                tasks_metric = {
                    eval_task + "_" + k: v for k, v in output.metrics.items()
                }
            for key in sorted(tasks_metric.keys()):
                logger.info(f"  {key} = {tasks_metric[key]}")
            results.update(tasks_metric)
            wandb.log(tasks_metric)
            reset_config(self.model, model_config)

        # Computes the average metrics across all the tasks without their corresponding losses.
        metrics = [results[key] for key in results.keys() if "loss" not in key]
        results["eval_average_metrics"] = np.mean(metrics)
        logger.info(f"  average_metrics = {np.mean(metrics)}")
        if is_test:
            wandb.log({"test/eval_average_metrics": np.mean(metrics)})
        else:
            wandb.log({"eval_average_metrics": np.mean(metrics)})
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, results
        )
        return results