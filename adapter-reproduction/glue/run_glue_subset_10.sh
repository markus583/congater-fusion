MODEL_NAME=bert-base-uncased
GPU_ID=0
SEED=$GPU_ID
NEW_SEED=-1

while getopts ":g:s:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG"
    ;;
    s) NEW_SEED="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
        exit 1
    ;;
  esac
done

# if NEW_SEED is not -1, use it as new seed. otherwise, use GPU_ID as seed
# shellcheck disable=SC2086
if [ $NEW_SEED -eq -1 ]; then
  SEED=$GPU_ID
else
  SEED=$NEW_SEED
fi

echo $SEED

for TASK in mrpc rte sst2 cola stsb qnli mnli qqp; do
  for SEED in $SEED; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi

    for TRAIN_PCT in 10 25 50; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python run_glue_fix_eval.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 0 \
        --learning_rate 1e-4 \
        --num_train_epochs 30 \
        --train_adapter \
        --adapter_config pfeiffer \
        --output_dir runs/st-a/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
        --logging_strategy epoch \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --early_stopping True \
        --early_stopping_patience 5 \
        --load_best_model_at_end True \
        --metric_for_best_model $EVAL_METRIC \
        --report_to wandb \
        --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED \
        --seed $SEED \
        --max_train_pct $TRAIN_PCT
    done
  done
done