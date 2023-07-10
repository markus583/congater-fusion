RUN_NAME=MTA-MULTI-GSG2
TASKS=(cb copa wsc rte mrpc cola wic boolq stsb sst2 multirc qnli mnli qqp record)

MODEL_NAME=roberta-base
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

# if no seeds are specified, use default seeds 0 to 2
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2)
fi


for SEED in "${SEEDS[@]}"; do
  echo $RUN_NAME
  echo $SEED, ${SEEDS[@]}
  echo ${TASKS[@]}

  CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run_multi.py \
    --model_name_or_path $MODEL_NAME \
    --max_seq_length 128 \
    --tasks ${TASKS[@]} \
    --eval_tasks ${TASKS[@]} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --learning_rate 1e-4 \
    --output_dir ../../runs/$RUN_NAME/$MODEL_NAME/$SEED \
    --max_steps 262144 \
    --logging_strategy steps \
    --logging_steps 20 \
    --save_strategy steps \
    --save_steps 8192 \
    --evaluation_strategy steps \
    --eval_steps 8192 \
    --early_stopping \
    --early_stopping_patience 5 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss_mean \
    --greater_is_better False \
    --report_to wandb \
    --run_name $RUN_NAME-$MODEL_NAME-$SEED \
    --seed $SEED \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --train_adapter \
    --freeze_base_model True \
    --separate_task_adapter False \
    --fp16 \

    rm -rf ../../runs/$RUN_NAME/$MODEL_NAME/$SEED
done
