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
                "f1_MEAN",
                "f1_STD",
                "exact_match_MEAN",
                "exact_match_STD",
                "combined_score_MEAN",
                "combined_score_STD",
            ]
        )

        tasks = tasks2list[task_list]

        for task in tasks:
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
                eval_results_dirs = [
                    os.path.join(d, "eval_results.json")
                    for d in seed_dirs
                    if os.path.isfile(os.path.join(d, "eval_results.json"))
                ]
                test_results_dirs = [
                    os.path.join(d, "test_results.json")
                    for d in seed_dirs
                    if os.path.isfile(os.path.join(d, "test_results.json"))
                ]
                for res in test_results_dirs:
                    eval_results_dirs.append(res)
                # take, if existing from json: eval_accuracy, eval_pearson, eval_spearmanr, eval_matthews_correlation
                # take, if existing from json: eval_accuracy, eval_pearson, eval_spearmanr, eval_matthews_correlation
                all_metrics = {}
                for eval_result in eval_results_dirs:
                    with open(eval_result) as f:
                        metrics = json.load(f)
                    current_metrics = {}
                    for metric in [
                        "accuracy",
                        "accuracy_mm",
                        "pearson",
                        "spearmanr",
                        "matthews_correlation",
                        "f1",
                        "exact_match",
                        "combined_score",
                    ]:
                        if "test_" + metric in metrics:
                            current_metrics[metric] = metrics["test_" + metric]
                        elif "eval_" + metric in metrics:
                            current_metrics[metric] = metrics["eval_" + metric]

                    # get seed
                    seed = int(eval_result.split("/")[-2])
                    # update metrics
                    if not (task == "wsc" and current_metrics["accuracy"] < 0.56):
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
                    if task in ["multirc", "record"]:
                        if all_metrics[seed]["f1"] > all_metrics[best_seed]["f1"]:
                            best_seed = seed
                    elif task == "cola":
                        if (
                            all_metrics[seed]["matthews_correlation"]
                            > all_metrics[best_seed]["matthews_correlation"]
                        ):
                            best_seed = seed
                    elif task == "stsb":
                        if (
                            all_metrics[seed]["pearson"]
                            > all_metrics[best_seed]["pearson"]
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
                    "accuracy_mm",
                    "pearson",
                    "spearmanr",
                    "matthews_correlation",
                    "f1",
                    "exact_match",
                    "combined_score"
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
                        "n_runs": len(all_metrics),
                    },
                    ignore_index=True,
                )

        # if no train_pct 10, 25, 50, 100: add empty row, task mrpc, n_runs 0
        for train_pct in ["10", "25", "50", "100"]:
            if train_pct not in df["train_pct"].unique():
                if task_list == "GLUE":
                    df = df.append(
                        {
                            "task": "mrpc",
                            "train_pct": train_pct,
                            "n_runs": 0,
                            "best_seed": None,
                            "seeds": [],
                            "accuracy_MEAN": None,
                            "accuracy_STD": None,
                            "accuracy_mm_MEAN": None,
                            "accuracy_mm_STD": None,
                            "pearson_MEAN": None,
                            "pearson_STD": None,
                            "spearmanr_MEAN": None,
                            "spearmanr_STD": None,
                            "matthews_correlation_MEAN": None,
                            "matthews_correlation_STD": None,
                            "f1_MEAN": None,
                            "f1_STD": None,
                        },
                        ignore_index=True,
                    )
                else:
                    df = df.append(
                        {
                            "task": "cb",
                            "train_pct": train_pct,
                            "n_runs": 0,
                            "best_seed": None,
                            "seeds": [],
                            "accuracy_MEAN": None,
                            "accuracy_STD": None,
                            "accuracy_mm_MEAN": None,
                            "accuracy_mm_STD": None,
                            "pearson_MEAN": None,
                            "pearson_STD": None,
                            "spearmanr_MEAN": None,
                            "spearmanr_STD": None,
                            "matthews_correlation_MEAN": None,
                            "matthews_correlation_STD": None,
                            "f1_MEAN": None,
                            "f1_STD": None,
                        },
                        ignore_index=True,
                    )
        df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    print("DONE!")


if __name__ == "__main__":
    extract_statistics()
