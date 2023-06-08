RUN_NAME=ST-A-FUSION-GSG-FP16-3_adp
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
  SEEDS=(0 1 2 3 4 5 6 7 8 9)
fi


for TASK in cb copa wsc rte mrpc wic stsb boolq sst2 qnli qqp mnli; do
  for SEED in "${SEEDS[@]}"; do
    if [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi

    # if task copa: gradient_checkpointing
    # due to memory constraints
    if [ $TASK = "copa" ]; then
      GRADIENT_CHECKPOINTING="--gradient_checkpointing"
    else
      GRADIENT_CHECKPOINTING=""
    fi

    # these tasks only run with seeds 0 to 4
    if [ $SEED -gt 4 ] && [ $TASK = "sst2" -o $TASK = "qnli" -o $TASK = "qqp" -o $TASK = "mnli" ]; then
      echo "Skipping $TASK with seed $SEED"
      continue
    fi

    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $TRAIN_PCT

      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --train_fusion \
        --fusion_type dynamic \
        --fusion_load_dir cf_config_GSG-3_only.json \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 0 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        --output_dir ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
        --logging_strategy steps \
        --logging_steps 20 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --early_stopping True \
        --early_stopping_patience 5 \
        --load_best_model_at_end True \
        --metric_for_best_model $EVAL_METRIC \
        --report_to wandb \
        --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME \
        --seed $SEED \
        --max_train_pct $TRAIN_PCT \
        --overwrite_output_dir \
        --fp16 \
        $GRADIENT_CHECKPOINTING

        rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
    done
  done
done
