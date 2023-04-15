MODEL_NAME=bert-base-uncased
GPU_ID=0
SEEDS=-1  # set the default seed(s) to be the same as GPU_ID

while getopts ":g:s:" opt; do
  # shellcheck disable=SC2220
  case $opt in
    g) GPU_ID="$OPTARG"
    ;;
    s) SEEDS="$OPTARG"
    ;;
  esac
done


IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"  # split SEEDS into an array


# if the seed is not specified, use the GPU ID as the seed
if [ "${SEED_ARRAY[0]}" -eq -1 ]; then
  SEED_ARRAY[0]=$GPU_ID
fi

for TASK in mrpc rte sst2 cola stsb qnli mnli qqp; do
  for SEED in "${SEED_ARRAY[@]}"; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi
    echo $SEED

    for TRAIN_PCT in 10 25 50 100; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python run_glue_FULL.py \
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
        --output_dir runs/full/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
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