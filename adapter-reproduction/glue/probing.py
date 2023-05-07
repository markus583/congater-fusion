import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# runs are located in:
# runs/st-a/$TASK_NAME/$MODEL_NAME/$SEED/checkpoint-xxxxx


# do this for every TASK_NAME and MAX_SEQ_LEN
# calculate mean and standard deviation of WANTED_NUMB

MODEL_NAME = "bert-base-uncased"

DIR_NAMES = [
    # "PROBE/ct_0-a-RELU-PLUS-LN_BEFORE",
    "PROBE/ct_0-a-RELUcopy",
    "PROBE/ct_2-a-RELU-PLUS-gate_adapter",
    "PROBE/ct_2-a-RELU-PLUS-LN_BEFORE-gate_adapter",
    "PROBE/ct_3-a-V3",
    "PROBE/ct_3-a-V4",
]

for dir_name in DIR_NAMES:
    for OMEGA in ["00", "01", "03", "05", "07", "09", "1"]:
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
        for task in ["rte", "mnli", "qqp", "sst2", "mrpc", "qnli", "stsb", "cola"]:
            # get all directories in subfolder
            subfolder = os.path.join("..", "..", "src", "runs", dir_name, OMEGA,task, MODEL_NAME)
            if not os.path.isdir(subfolder):
                continue
            subfolder_content = os.listdir(subfolder)
            # filter: only 10, 25, 50 of subfolder_content remain
            TRAIN_PCT_LIST = [
                x for x in ["10", "25", "50", "100"] if x in subfolder_content
            ]
            for TRAIN_PCT in TRAIN_PCT_LIST:
                subfolder = os.path.join(
                    "..", "..", "src", "runs", dir_name, OMEGA, task, MODEL_NAME, TRAIN_PCT
                )
                subfolder_content = os.listdir(subfolder)
                if len(subfolder_content) == 0:
                    continue
                # get all directories with seed
                seed_dirs = sorted([
                    os.path.join(subfolder, d)
                    for d in subfolder_content
                    if os.path.isdir(os.path.join(subfolder, d))
                ])
                # get all directories with eval_results.json
                # eval_results_dirs = [
                #     os.path.join(d, "eval_results.json")
                #     for d in seed_dirs
                #     if os.path.isfile(os.path.join(d, "eval_results.json"))
                # ]
                eval_results_dirs = []
                test_results_dirs = [
                    os.path.join(d, "test_results.json")
                    for d in seed_dirs
                    if os.path.isfile(os.path.join(d, "test_results.json"))
                ]
                for res in test_results_dirs:
                    eval_results_dirs.append(res)
                eval_results_dirs = [sorted(eval_results_dirs)[0]]
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
                    if task == "cola":
                        if (
                            all_metrics[seed]["matthews_correlation"]
                            > all_metrics[best_seed]["matthews_correlation"]
                        ):
                            best_seed = seed
                    elif task == "stsb":
                        if all_metrics[seed]["pearson"] > all_metrics[best_seed]["pearson"]:
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
                        "matthews_correlation_MEAN": mean_metrics["matthews_correlation"],
                        "matthews_correlation_STD": std_metrics["matthews_correlation"],
                        "n_runs": len(all_metrics),
                    },
                    ignore_index=True,
                )

        output_file = "results/probe/" + dir_name.split("/")[1] + "_" + OMEGA + ".csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")



print("DONE!")
