MODEL_NAME=bert-base-uncased
NR=2

for TASK in mrpc rte sst2 cola stsb qnli mnli qqp; do
  for SEED in $NR; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi
    CUDA_VISIBLE_DEVICES=$NR python run_glue_FULL.py \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK \
      --max_seq_length 128 \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 0 \
      --learning_rate 2e-5 \
      --num_train_epochs 30 \
      --output_dir runs/full/$TASK/$MODEL_NAME/$SEED \
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