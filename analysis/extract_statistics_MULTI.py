import json
import os
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# runs are located in:
# runs/full/$TASK_NAME/$MODEL_NAME/$SEED/checkpoint-xxxxx


tasks2list = {}

tasks2list["GLUE"] = [
    "rte",
    "mrpc",
    "cola",
    "stsb",
    "sst2",
    "qnli",
    "qqp",
    "mnli",
]

tasks2list["SUPERGLUE"] = ["cb", "copa", "wsc", "wic", "boolq", "multirc", "record"]

tasks2list["GSG2"] = tasks2list["GLUE"] + tasks2list["SUPERGLUE"]

# do this for every TASK_NAME and MAX_SEQ_LEN
# calculate mean and standard deviation of WANTED_NUMB
def extract_statistics(OUT_FILES, DIR_NAMES, MODEL_NAME, task_list):
    for output_file, dir_name in zip(OUT_FILES, DIR_NAMES):
        df = pd.DataFrame(
            columns=[
                "task",
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
                "f1_MEAN",
                "f1_STD",
                "exact_match_MEAN",
                "exact_match_STD",
                "combined_score_MEAN",
                "combined_score_STD",
            ]
        )

        tasks = tasks2list[task_list]
        for seed in range(10):
            subfolder = os.path.join("..", "src", "training", "runs", dir_name)  # , MODEL_NAME, seed)
            if not os.path.isdir(subfolder):
                continue
            test_results = os.path.join(subfolder, "test_results.json")
            if not os.path.isfile(test_results):
                continue
            results = json.load(open(test_results))
            all_metrics = {}

            for task in tasks:
                # get keys with task
                # filter out loss, runtime, samples, steps
                metric_keys = [
                    k
                    for k in results.keys()
                    if task in k and "loss" not in k and "runtime" not in k and "samples" not in k and "steps" not in k
                ]
                for metric in metric_keys:
                    if task not in all_metrics:
                        all_metrics[task] = {}
                    if "combined_score" in metric:
                        metric_name = metric.split("_")[-2] + "_" + metric.split("_")[-1]
                    else:
                        metric_name = metric.split("_")[-1]
                    if metric_name not in all_metrics[task]:
                        all_metrics[task][metric_name] = []
                    all_metrics[task][metric_name].append(results[metric])    
                
        
        for task in all_metrics.keys():
            mean_metrics = {}
            std_metrics = {}
            for metric in [
                "accuracy",
                "accuracy_mm",
                "pearson",
                "spearmanr",
                "matthews_correlation",
                "f1",
                "exact_match",
                "combined_score"
            ]:
                
                if metric not in all_metrics[task]:
                    mean_metrics[metric] = None
                    std_metrics[metric] = None
                    continue
                mean_metrics[metric] = np.mean(all_metrics[task][metric])
                std_metrics[metric] = np.std(all_metrics[task][metric])
                if task == "stsb" and metric == "pearson":
                    best_seed = np.argmax(all_metrics[task][metric])
                    n_runs = len(all_metrics[task][metric])
                elif task == "cola" and metric == "matthews_correlation":
                    best_seed = np.argmax(all_metrics[task][metric])
                    n_runs = len(all_metrics[task][metric])
                elif task in ["multirc", "record"] and metric == "f1":
                    best_seed = np.argmax(all_metrics[task][metric])
                    n_runs = len(all_metrics[task][metric])
                elif metric == "accuracy":
                    best_seed = np.argmax(all_metrics[task][metric])
                    n_runs = len(all_metrics[task][metric])
                else:
                    best_seed = None
                    n_runs = None
                best_seed = np.argmax(all_metrics[task][metric])
                # add to dataframe
            df = df.append(
                {
                    "task": task,
                    "n_runs": n_runs,
                    "best_seed": best_seed,
                    "seeds": None,
                    "accuracy_MEAN": mean_metrics["accuracy"],
                    "accuracy_STD": std_metrics["accuracy"],
                    "accuracy_mm_MEAN": mean_metrics["accuracy_mm"],
                    "accuracy_mm_STD": std_metrics["accuracy_mm"],
                    "pearson_MEAN": mean_metrics["pearson"],
                    "pearson_STD": std_metrics["pearson"],
                    "spearmanr_MEAN": mean_metrics["spearmanr"],
                    "spearmanr_STD": std_metrics["spearmanr"],
                    "matthews_correlation_MEAN": mean_metrics[
                        "matthews_correlation"
                    ],
                    "matthews_correlation_STD": std_metrics["matthews_correlation"],
                    "f1_MEAN": mean_metrics["f1"],
                    "f1_STD": std_metrics["f1"],
                    "exact_match_MEAN": mean_metrics["exact_match"],
                    "exact_match_STD": std_metrics["exact_match"],
                    "combined_score_MEAN": mean_metrics["combined_score"],
                    "combined_score_STD": std_metrics["combined_score"],
                },
                ignore_index=True,
            )

        df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    print("DONE!")


if __name__ == "__main__":
    MODEL_NAME = "roberta-base"
    TASKS = "GSG2"
    BASE_DIR = BASE_DIR = f"{TASKS}-{MODEL_NAME}/results/"
    OUT_FILE = "MULTI-TEST.csv"
    OUT_FILES = [BASE_DIR + OUT_FILE]
    DIR_NAME = ["TEST"]
    
    extract_statistics(OUT_FILES, DIR_NAME, MODEL_NAME, TASKS)
