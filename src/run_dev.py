import logging
import os
import random
import sys

# os.chdir(os.path.dirname(os.path.realpath(__file__)))

# sys.path.append("/home/markus-frohmann/congater-fusion/adapter-transformers/src")

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import get_args
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS, TASKS
from training.get_trainer_dev import get_trainer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "THESIS_st-a"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL "] = "true"
logger = logging.getLogger(__name__)


def train_fn(trainer, training_args, last_checkpoint=None):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def evaluate_fn(trainer, data_args, dataset):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    test_datasets = [dataset.test_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = dataset.test_dataset_mm
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
            valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
        test_datasets.append(valid_mm_dataset)
        combined = {}

    for ds, task in zip(test_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix="test")

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(ds)
        )
        metrics["test_samples"] = min(max_eval_samples, len(ds))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics(
            "test", combined if task is not None and "mnli" in task else metrics
        )


def detect_last_checkpoint(training_arguments: transformers.TrainingArguments) -> str:
    checkpoint = None
    if (
        os.path.isdir(training_arguments.output_dir)
        and training_arguments.do_train
        and not training_arguments.overwrite_output_dir
    ):
        checkpoint = get_last_checkpoint(training_arguments.output_dir)
        if checkpoint is None and len(os.listdir(training_arguments.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_arguments.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif (
            checkpoint is not None and training_arguments.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return checkpoint


def setup_logging(training_args: transformers.TrainingArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def main() -> None:
    args = get_args()
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        congater_args,
    ) = args

    if adapter_args.train_adapter:
        if adapter_args.adapter_config == "pfeiffer":
            WANDBPROJECT = "THESIS_st-a"
        else:
            WANDBPROJECT = "THESIS_ct_1-a"
    elif fusion_args.train_fusion:
        if congater_args.congosition_type:
            WANDBPROJECT = "THESIS_congosition"
        else:
            WANDBPROJECT = "THESIS_st-a-fusion"
    elif not adapter_args.train_adapter and not fusion_args.train_fusion:
        WANDBPROJECT = "THESIS_full"
    else:
        raise NotImplementedError
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_LOG_MODEL "] = "true"

    os.environ["WANDB_PROJECT"] = WANDBPROJECT
    os.environ[
        "WANDB_NAME"
    ] = f"{data_args.task_name}-{model_args.model_name_or_path}-{data_args.max_train_pct}-{training_args.seed}"

    setup_logging(training_args)

    if not data_args.dataset_name:
        # infer dataset and task from task_name
        if data_args.task_name.lower() in GLUE_DATASETS:
            data_args.dataset_name = "glue"
        elif data_args.task_name.lower() in SUPERGLUE_DATASETS:
            data_args.dataset_name = "superglue"
        else:
            raise ValueError(
                f"Task {data_args.task_name} not found. Should be one of {TASKS}"
            )
    print("dataset_name", data_args.dataset_name)
    print("task_name", data_args.task_name)

    if not data_args.eval_adapter:
        last_checkpoint = detect_last_checkpoint(training_arguments=training_args)
    else:
        last_checkpoint = None

    set_seed(training_args.seed)

    trainer, model, dataset, adapter_setup = get_trainer(args=args)

    if training_args.do_train:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {dataset.train_dataset[index]}."
            )

        train_fn(trainer, training_args, last_checkpoint)

    # save adapter
    print(adapter_args)
    if fusion_args.train_fusion:
        logger.info("Saving Fusion.")

        if congater_args.congosition_type is not None:
            model.save_congosition(training_args.output_dir, ",".join(adapter_setup[0]))
        else:
            model.save_adapter_fusion(
                training_args.output_dir, ",".join(adapter_setup[0])
            )
    elif adapter_args.train_adapter:
        logger.info("Saving adapter.")
        if data_args.source_task:
            model.save_adapter(training_args.output_dir, data_args.source_task)
        elif not congater_args.congosition_type:
            model.save_adapter(training_args.output_dir, data_args.task_name)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        evaluate_fn(trainer, data_args, dataset)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_args"] = data_args.task_name
        if data_args.dataset_name.lower() == "glue":
            kwargs["dataset_tags"] = data_args.dataset_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"
        elif data_args.dataset_name.lower() == "superglue":
            kwargs["dataset_tags"] = data_args.dataset_name
            kwargs["dataset"] = f"SuperGLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    # if model_name_or_path not given, set params
    print(sys.argv)
    sys.path.append("/home/markus-frohmann/congater-fusion/adapter-transformers/src")
    if "--model_name_or_path" not in sys.argv:
        sys.argv += [
            "--model_name_or_path",
            "roberta-base",
            "--task_name",
            "record",
            "--dataset_name",
            "superglue",
            "--max_seq_length",
            "128",
            "--do_train",
            "--do_eval",
            "--max_seq_length",
            "128",
            "--per_device_train_batch_size",
            "32",
            "--per_device_eval_batch_size",
            "32",
            "--learning_rate",
            "5e-3",
            "--num_train_epochs",
            "30",
            "--train_adapter",
            "True",
            # "--train_fusion",
            # "--fusion_load_dir",
            # "scripts/st-c5_fusion/cf_config_GSG.json",
            "--output_dir",
            "runs/TEST",
            # "--eval_adapter",
            # "True",
            "--logging_strategy",
            "steps",
            "--logging_steps",
            "10",
            "--evaluation_strategy",
            "steps",
            "--save_strategy",
            "epoch",
            "--early_stopping",
            "False",
            "--early_stopping_patience",
            "5",
            # "--load_best_model_at_end",
            # "True",
            # "--metric_for_best_model",
            # "eval_accuracy",
            "--report_to",
            "wandb",
            "--run_name",
            "i0-rte-TEST",
            "--max_train_pct",
            "10",
            # "--max_train_samples",
            # "100",
            "--seed",
            "2",
            "--overwrite_output_dir",
            "--no_cuda",
            # "--max_steps",
            # "10",
            "--adapter_config",
            "pfeiffer",
            "--omega",
            "0.5",
            "--debug_congater",
            "True",
            # "--congosition_type",
            # "full_fusion_bottleneck-r2",
            # "--fusion_type",
            # "full_fusion_bottleneck-r2",
            # "--learn_omega",
            # "False",
            # "--train_probing_head",
            # "True",
            # "--omega_grid",
            # "{'mnli': 0.0, 'qqp': 0.0}"
            # "--fp16",
        ]
    main()
