MODEL_NAME=bert-base-uncased
TASK=mrpc
SEED=0


CUDA_VISIBLE_DEVICES=2 python run_fusion_glue_ST-A.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK \
  --max_seq_length 128 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 0 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --output_dir runs/st-a-fusion/$TASK/$MODEL_NAME/$SEED \
  --logging_strategy epoch \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --early_stopping True \
  --early_stopping_patience 5 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_accuracy \
  --report_to wandb \
  --run_name $TASK-$MODEL_NAME-$SEED \
  --seed $SEED
