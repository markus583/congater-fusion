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

# if no seeds are specified, use GPU_ID as seed
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=($GPU_ID)
fi

for TASK in qqp; do
  for SEED in "${SEEDS[@]}"; do
    if [ $TASK = "cola" ]; then
        EVAL_METRIC="eval_matthews_correlation"
    elif [ $TASK = "stsb" ]; then
        EVAL_METRIC="eval_pearson"
    else
        EVAL_METRIC="eval_accuracy"
    fi

    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $TRAIN_PCT

      for OMEGA in 00 01 03 05 07 09 1; do
        echo $OMEGA
        # change omega: 00 --> 0.0
        if [ $OMEGA = "00" ]; then
          omega=0.0
        elif 
          [ $OMEGA = "01" ]; then
          omega=0.1
        elif 
          [ $OMEGA = "03" ]; then
          omega=0.3
        elif 
          [ $OMEGA = "05" ]; then
          omega=0.5
        elif 
          [ $OMEGA = "07" ]; then
          omega=0.7
        elif 
          [ $OMEGA = "09" ]; then
          omega=0.9
        elif 
          [ $OMEGA = "1" ]; then
          omega=1.0
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
          --model_name_or_path $MODEL_NAME \
          --task_name $TASK \
          --max_seq_length 128 \
          --do_train \
          --do_eval \
          --eval_adapter True \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 32 \
          --dataloader_num_workers 0 \
          --learning_rate 1e-100 \
          --num_train_epochs 30 \
          --max_steps 1 \
          --train_adapter \
          --adapter_config congaterV5[omega=$omega] \
          --output_dir ../../runs/PROBE/$RUN_NAME/$OMEGA/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
          --logging_strategy steps \
          --logging_steps 20 \
          --save_strategy epoch \
          --evaluation_strategy epoch \
          --early_stopping True \
          --early_stopping_patience 5 \
          --load_best_model_at_end True \
          --metric_for_best_model $EVAL_METRIC \
          --report_to wandb \
          --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME-$OMEGA \
          --max_train_pct $TRAIN_PCT \
          --seed $SEED \
          --omega $omega
      done
    done
  done
done