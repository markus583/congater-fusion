import os

MAX_SEQ_LENGTH=128

tasks = ["cola", "stsb", "qnli", "wnli", "sst2", "mnli", "qqp", "mrpc", "rte"]
seeds = [101]

for task in tasks:
    for seed in seeds:
        if task == "cola":
            EVAL_METRIC="eval_matthews_correlation"
        elif task == "stsb":
            EVAL_METRIC="eval_pearson"
        else:
            EVAL_METRIC="eval_accuracy"
        os.system(f"CUDA_VISIBLE_DEVICES=0 python run_glue_fix_eval.py \
          --model_name_or_path bert-base-uncased \
          --task_name {task} \
          --max_seq_length {MAX_SEQ_LENGTH} \
          --do_train \
          --do_eval \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 32 \
          --dataloader_num_workers 4 \
          --learning_rate 1e-4 \
          --num_train_epochs 30 \
          --train_adapter \
          --adapter_config pfeiffer \
          --output_dir runs/st-a/{task}/{seed}-{MAX_SEQ_LENGTH} \
          --overwrite_output_dir \
          --logging_strategy epoch \
          --save_strategy epoch \
          --evaluation_strategy epoch \
          --early_stopping True \
          --early_stopping_patience 5 \
          --load_best_model_at_end True \
          --metric_for_best_model {EVAL_METRIC} \
          --report_to wandb \
          --run_name st-a-{task}-{seed}-{MAX_SEQ_LENGTH} \
          --seed {seed}")
