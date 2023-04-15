import json
import os

import numpy as np
import pandas as pd

# runs are located in:
# runs/st-a/$TASK_NAME/$MODEL_NAME/$SEED/checkpoint-xxxxx


# do this for every TASK_NAME and MAX_SEQ_LEN
# calculate mean and standard deviation of WANTED_NUMB

MODEL_NAME = "bert-base-uncased"

output_file = "results/ST-A-FUSION.csv"
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
    ]
)

for task in ["mnli", "qqp", "sst2", "mrpc", "rte", "qnli", "stsb", "cola"]:
    # get all directories in subfolder
    subfolder = os.path.join("runs", "st-a-fusion", task, MODEL_NAME)
    subfolder_content = os.listdir(subfolder)
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
    # take, if existing from json: eval_accuracy, eval_pearson, eval_spearmanr, eval_matthews_correlation
    all_metrics = []
    seeds = []
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
            if "eval_" + metric in metrics:
                current_metrics[metric] = metrics["eval_" + metric]
        # get seed
        seed = int(eval_result.split("/")[-2])
        seeds.append(seed)
        # update metrics
        all_metrics.append(current_metrics)
    # take mean and std of each metric
    mean_metrics = {}
    std_metrics = {}
    # sort seeds
    seeds.sort()
    best_seed = seeds[0]
    for seed in seeds:
        try:
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
                if all_metrics[seed]["accuracy"] > all_metrics[best_seed]["accuracy"]:
                    best_seed = seed
        except:
            continue
    for metric in [
        "accuracy",
        "accuracy_mm",
        "pearson",
        "spearmanr",
        "matthews_correlation",
    ]:
        metric_values = [float(m[metric]) for m in all_metrics if metric in m]
        mean_metrics[metric] = np.mean(metric_values)
        std_metrics[metric] = np.std(metric_values, ddof=1)
    # add to dataframe
    df = df.append(
        {
            "task": task,
            "best_seed": best_seed,
            "seeds": seeds,
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

df.to_csv(output_file, index=False)
