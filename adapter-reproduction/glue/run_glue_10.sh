MODEL_NAME=bert-base-uncased
NR=0

for TASK in rte; do
  for SEED in {4..9}; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi
    CUDA_VISIBLE_DEVICES=$NR python run_glue_fix_eval.py \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK \
      --max_seq_length 128 \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 4 \
      --learning_rate 1e-4 \
      --num_train_epochs 30 \
      --train_adapter \
      --adapter_config pfeiffer \
      --output_dir runs/st-a/$TASK/$MODEL_NAME/$SEED \
      --overwrite_output_dir \
      --logging_strategy epoch \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --early_stopping True \
      --early_stopping_patience 5 \
      --load_best_model_at_end True \
      --metric_for_best_model $EVAL_METRIC \
      --report_to wandb \
      --run_name $TASK-$MODEL_NAME-$SEED \
      --seed $SEED
  done
done