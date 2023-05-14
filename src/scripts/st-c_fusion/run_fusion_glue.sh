MODEL_NAME=bert-base-uncased
GPU_ID=0
SEEDS=()

while getopts ":g:s:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG"
    ;;
    s) SEEDS+=("$OPTARG")
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
        exit 1
    ;;
  esac
done

# if no seeds are specified, use GPU_ID as seed
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=($GPU_ID)
fi


for TASK in mrpc rte sst2 cola stsb qnli mnli qqp; do
  for SEED in "${SEEDS[@]}"; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi

    for TRAIN_PCT in 10 25 50 100; do
      echo $SEED
      echo $TASK
      echo $TRAIN_PCT
      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
        --model_name_or_path $MODEL_NAME \
        --dataset_name glue \
        --task_name $TASK \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --train_fusion \
        --fusion_load_dir cf_config_GLUE.json \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 0 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        --output_dir ../../runs/C-V2-fusion/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
        --logging_strategy steps \
        --logging_steps 100 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --early_stopping True \
        --early_stopping_patience 5 \
        --load_best_model_at_end True \
        --metric_for_best_model $EVAL_METRIC \
        --report_to wandb \
        --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-C-V2-FUSION-GLUE \
        --seed $SEED \
        --max_train_pct $TRAIN_PCT \
        --overwrite_output_dir
    done
  done
done
