import argparse
import os
from get_combinations import generate_value_combinations
import subprocess


RUN_NAME = "st-a"
MODEL_NAME = "bert-base-uncased"
GPU_ID = 0
SEEDS = []

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("-s", "--seeds", nargs="+", type=int, default=[], help="Seeds")
args = parser.parse_args()

GPU_ID = args.gpu_id
SEEDS = args.seeds

# if no seeds are specified, use default seeds 0 to 9
if len(SEEDS) == 0:
    SEEDS = list(range(10))

SOURCE_TASKS = ["SELF", "mnli"]
TARGET_TASKS = ["mrpc"]

# get all possible combinations for source tasks and omega values (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
# QQP: 0.0, MNLI: 0.0; QQP: 0.0, MNLI: 0.1, etc.
omegas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
source_task_combinations = generate_value_combinations(SOURCE_TASKS, omegas)
print(source_task_combinations)

for target_task in TARGET_TASKS:
    for SEED in SEEDS:
        print(f"SEED: {SEED}")
        for combo in source_task_combinations:
            current_combo = combo.copy()
            # replace key 'SELF' with target task
            current_combo[target_task] = current_combo.pop("SELF")  
            print(f"changing SELF to {target_task}")
        
            if target_task == "stsb":
                EVAL_METRIC = "eval_pearson"
            else:
                EVAL_METRIC = "eval_accuracy"

            # these tasks only run with seeds 0 to 4
            if (SEED > 1) and (target_task == "boolq" or target_task == "stsb"):
                print(f"Skipping {target_task} with seed {SEED}")
                continue

            if (SEED > 0) and (
                target_task == "mnli"
                or target_task == "qqp"
                or target_task == "qnli"
                or target_task == "sst2"
            ):
                print(f"Skipping {target_task} with seed {SEED}")
                continue

            if (SEED > 4) and (
                target_task == "cb"
                or target_task == "copa"
                or target_task == "mrpc"
                or target_task == "rte"
                or target_task == "wsc"
                or target_task == "wic"
            ):
                print(f"Skipping {target_task} with seed {SEED}")
                continue
            
            for TRAIN_PCT in [100]:
                print(RUN_NAME)
                print(SEED, SEEDS)
                print(f"target task: {target_task}")
                print(f"Source task combinations: {current_combo}")
                print(TRAIN_PCT)
                # parse combo into directory name
                combo_dir = ""
                for key, value in current_combo.items():
                    combo_dir += f"{key}-{value}/"
                # remove last underscore
                combo_dir = combo_dir[:-1]

                command = f"""
                CUDA_VISIBLE_DEVICES={GPU_ID} python ../../run_dev.py \
                    --model_name_or_path {MODEL_NAME} \
                    --task_name {target_task} \
                    --max_seq_length 128 \
                    --do_train \
                    --do_eval \
                    --train_probing_head True \
                    --per_device_train_batch_size 32 \
                    --per_device_eval_batch_size 32 \
                    --fusion_type omega_grid \
                    --dataloader_num_workers 0 \
                    --learning_rate 1e-4 \
                    --num_train_epochs 30 \
                    --train_adapter \
                    --adapter_config pfeiffer \
                    --output_dir ../../runs/OMEGA_GRID/{RUN_NAME}/{target_task}/{combo_dir}/{MODEL_NAME}/{TRAIN_PCT}/{SEED} \
                    --logging_strategy steps \
                    --logging_steps 50 \
                    --save_strategy epoch \
                    --evaluation_strategy epoch \
                    --early_stopping True \
                    --early_stopping_patience 5 \
                    --load_best_model_at_end True \
                    --metric_for_best_model {EVAL_METRIC} \
                    --run_name {target_task}-{MODEL_NAME}-{TRAIN_PCT}-{SEED}-{RUN_NAME}-{combo_dir} \
                    --max_train_pct {TRAIN_PCT} \
                    --seed {SEED} \
                    --overwrite_output_dir \
                    --congosition_type omega_grid \
                    --omega_grid "{current_combo}"
                """

                subprocess.run(command, shell=True, check=True)
                try:
                    stdout, stderr = subprocess.communicate()
                    # stop entire loop in case of keyboard interrupt
                    if stdout == KeyboardInterrupt:
                        raise KeyboardInterrupt
                except:
                    pass

                checkpoint_path = f"../../runs/OMEGA_GRID/{RUN_NAME}/{combo_dir}/{target_task}/{MODEL_NAME}/{TRAIN_PCT}/{SEED}/checkpoint*"
                os.system(f"rm -rf {checkpoint_path}")
