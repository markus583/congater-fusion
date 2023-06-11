RUN_NAME=C-V5

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

SOURCE_TASKS=("qqp" "mnli")
TARGET_TASKS=("cb" "copa" "wsc" "rte" "mrpc")

for target_task in "${TARGET_TASKS[@]}"; do 
  for SEED in "${SEEDS[@]}"; do
    if [ $target_task = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi

    # these tasks only run with seeds 0 to 4
    if [ $SEED -gt 1 ] && [ $target_task = "boolq" -o $target_task = "stsb" ]; then
        echo "Skipping $target_task + $source_task with seed $SEED"
        continue
    fi

    if [ $SEED -gt 0 ] && [ $target_task = "mnli" -o $target_task = "qqp" -o $target_task = "qnli" -o $target_task = "sst2" -o $target_task = "wic" ]; then
        echo "Skipping $target_task + $source_task with seed $SEED"
        continue
    fi

    if [ $SEED -gt 4 ] && [ $target_task = "cb" -o $target_task = "copa" -o $target_task = "mrpc" -o $target_task = "rte" -o $target_task = "wsc" ]; then
        echo "Skipping $target_task + $source_task with seed $SEED"
        continue
    fi

    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo "target task: $target_task"
      echo $TRAIN_PCT

        for OMEGA_1 in 00 01 03 05 07 09 1; do
          echo $OMEGA_1
          # change omega: 00 --> 0.0
          if [ $OMEGA_1 = "00" ]; then
            omega_1=0.0
          elif 
            [ $OMEGA_1 = "01" ]; then
            omega_1=0.1
          elif 
            [ $OMEGA_1 = "03" ]; then
            omega_1=0.3
          elif 
            [ $OMEGA_1 = "05" ]; then
            omega_1=0.5
          elif 
            [ $OMEGA_1 = "07" ]; then
            omega_1=0.7
          elif 
            [ $OMEGA_1 = "09" ]; then
            omega_1=0.9
          elif 
            [ $OMEGA_1 = "1" ]; then
            omega_1=1.0
          fi

          CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
              --model_name_or_path $MODEL_NAME \
              --task_name $target_task \
              --max_seq_length 128 \
              --do_train \
              --do_eval \
              --eval_adapter True \
              --train_probing_head True \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --dataloader_num_workers 0 \
              --learning_rate 1e-4 \
              --num_train_epochs 30 \
              --train_adapter \
              --adapter_config pfeiffer \
              --output_dir ../../runs/PPROBE/$RUN_NAME/$OMEGA_1/$source_task/$target_task/$MODEL_NAME/$TRAIN_PCT/$SEED \
              --logging_strategy steps \
              --logging_steps 50 \
              --save_strategy epoch \
              --evaluation_strategy epoch \
              --early_stopping True \
              --early_stopping_patience 5 \
              --load_best_model_at_end True \
              --metric_for_best_model $EVAL_METRIC \
              --report_to wandb \
              --run_name $target_task-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME-$OMEGA_1-$source_task \
              --max_train_pct $TRAIN_PCT \
              --seed $SEED \
              --overwrite_output_dir \
              --omega $omega \
              --source_task $source_task \

              rm -rf ../../runs/PPROBE/$RUN_NAME/$OMEGA_1/$source_task/$target_task/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
        done
      done
    done
  done
done