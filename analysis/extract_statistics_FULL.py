import json
import os
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# runs are located in:
# runs/full/$TASK_NAME/$MODEL_NAME/$SEED/checkpoint-xxxxx


# do this for every TASK_NAME and MAX_SEQ_LEN
# calculate mean and standard deviation of WANTED_NUMB

MODEL_NAME = "bert-base-uncased"

OUT_FILES = ["results/FULL.csv"]
DIR_NAMES = ["full"]
for output_file, dir_name in zip(OUT_FILES, DIR_NAMES):
    df = pd.DataFrame(
        columns=[
            "task",
            "train_pct",
            "n_runs",
            "best_seed",
            "seeds",
            "accuracy_MEAN",
            "accuracy_STD",
            "accuracy_mm_MEAN",
            "accuracy_mm_STD",
            "pearson_MEAN",
            "pearson_STD",
            "spearmanr_MEAN",
            "spearmanr_STD",
            "matthews_correlation_MEAN",
            "matthews_correlation_STD",
        ]
    )

    for task in ["cb", "copa", "wsc", "multirc", "wic", "boolq", "record"]:
        # get all directories in subfolder
        subfolder = os.path.join("..", "src", "runs", dir_name, task, MODEL_NAME)
        if not os.path.isdir(subfolder):
            continue
        subfolder_content = os.listdir(subfolder)
        # filter: only 10, 25, 50 of subfolder_content remain
        TRAIN_PCT_LIST = [
            x for x in ["10", "25", "50", "100"] if x in subfolder_content
        ]
        for TRAIN_PCT in TRAIN_PCT_LIST:
            subfolder = os.path.join(
                "..", "src", "runs", dir_name, task, MODEL_NAME, TRAIN_PCT
            )
            subfolder_content = os.listdir(subfolder)
            if len(subfolder_content) == 0:
                continue
            # get all directories with seed
            seed_dirs = [
                os.path.join(subfolder, d)
                for d in subfolder_content
                if os.path.isdir(os.path.join(subfolder, d))
            ]
            # get all directories with eval_results.json
            test_results_dirs = [
                os.path.join(d, "test_results.json")
                for d in seed_dirs
                if os.path.isfile(os.path.join(d, "test_results.json"))
            ]
            # take, if existing from json: eval_accuracy, eval_pearson, eval_spearmanr, eval_matthews_correlation
            all_metrics = {}
            for eval_result in test_results_dirs:
                with open(eval_result) as f:
                    metrics = json.load(f)
                current_metrics = {}
                for metric in [
                    "accuracy",
                    "f1",
                ]:
                    if "test_" + metric in metrics:
                        current_metrics[metric] = metrics["test_" + metric]
                    elif "eval_" + metric in metrics:
                        current_metrics[metric] = metrics["eval_" + metric]
                # get seed
                seed = int(eval_result.split("/")[-2])
                # update metrics
                all_metrics[seed] = current_metrics
            # take mean and std of each metric
            mean_metrics = {}
            std_metrics = {}
            # sort seeds
            if len(all_metrics) == 0:
                continue
            best_seed = sorted(list(all_metrics.keys()))[0]
            for seed in all_metrics.keys():
                if len(all_metrics[seed]) == 0:
                    continue
                elif len(all_metrics[best_seed]) == 0:
                    continue
                if task  in ["multirc", "record"]:
                    if (
                        all_metrics[seed]["f1"]
                        > all_metrics[best_seed]["f1"]
                    ):
                        best_seed = seed
                else:
                    if (
                        all_metrics[seed]["accuracy"]
                        > all_metrics[best_seed]["accuracy"]
                    ):
                        best_seed = seed
            for metric in [
                "accuracy",
                "f1",
            ]:
                metric_values = [
                    float(m[metric]) for m in all_metrics.values() if metric in m
                ]
                mean_metrics[metric] = np.mean(metric_values)
                std_metrics[metric] = np.std(metric_values, ddof=1)
            # add to dataframe
            df = df.append(
                {
                    "task": task,
                    "train_pct": TRAIN_PCT,
                    "best_seed": best_seed,
                    "seeds": list(all_metrics.keys()),
                    "accuracy_MEAN": mean_metrics["accuracy"],
                    "accuracy_STD": std_metrics["accuracy"],
                    "f1_MEAN": mean_metrics["f1"],
                    "f1_STD": std_metrics["f1"],
                    "n_runs": len(all_metrics),
                },
                ignore_index=True,
            )

    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")



print("DONE!")