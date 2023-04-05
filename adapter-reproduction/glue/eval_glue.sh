MAX_SEQ_LENGTH=256

for TASK in mrpc; do
  for SEED in 1001; do
   CUDA_VISIBLE_DEVICES=0 python run_glue.py \
      --model_name_or_path bert-base-uncased \
      --load_adapter runs/st-a/mrpc/1001-256 \
      --task_name $TASK \
      --max_seq_length $MAX_SEQ_LENGTH \
      --do_eval \
      --do_train \
      --train_adapter \
      --num_train_epochs 1 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 4 \
      --adapter_config pfeiffer \
      --output_dir runs/st-a/$TASK/$SEED-$MAX_SEQ_LENGTH \
      --overwrite_output_dir \
      --logging_strategy epoch \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --early_stopping True \
      --early_stopping_patience 5 \
      --load_best_model_at_end True \
      --metric_for_best_model eval_accuracy \
      --report_to wandb \
      --run_name st-a-$TASK-$SEED-$MAX_SEQ_LENGTH \
      --seed $SEED
  done
done
