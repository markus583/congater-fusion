LR=5e-4
CONFIG=param_direct_clamp_avg-init-difflr
RUN_NAME=st-a-congosition_naive-$CONFIG-lr$LR

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

# if no seeds are specified, use default seeds 0 to 9
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2 3 4)
fi


for TASK in cb copa wsc rte mrpc wic stsb boolq sst2 qnli qqp mnli; do
  for SEED in "${SEEDS[@]}"; do
    # these tasks only run with seeds 0 to 4
    if [ $SEED -gt 2 ] && [ $TASK = "sst2" -o $TASK = "qnli" -o $TASK = "qqp" -o $TASK = "mnli" ]; then
      echo "Skipping $TASK with seed $SEED"
      continue
    fi
    
    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $TRAIN_PCT

      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run_dev.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --train_fusion \
        --fusion_load_dir af_config_GSG.json \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 0 \
        --learning_rate $LR \
        --num_train_epochs 30 \
        --output_dir ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
        --logging_strategy epoch \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --early_stopping True \
        --early_stopping_patience 15 \
        --load_best_model_at_end True \
        --lr_scheduler_type linear \
        --report_to wandb \
        --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME \
        --seed $SEED \
        --max_train_pct $TRAIN_PCT \
        --overwrite_output_dir \
        --fp16 \
        --congosition_type $CONFIG \
        --fusion_type $CONFIG \

        rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
    done
  done
done

# metric_for_best_model: eval_
# --warmup_ratio 0.1 \
