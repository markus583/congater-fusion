from itertools import combinations
import os
import warnings
import shutil

from PProbing import probing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

probing()


def sort_tasks(task):
    # cb, copa, wsc, rte, mrpc, wic, stsb, boolq, sst2, qnli, qqp, mnli
    if task == "cb":
        return 1
    elif task == "copa":
        return 2
    elif task == "wsc":
        return 3
    elif task == "rte":
        return 4
    elif task == "mrpc":
        return 5
    elif task == "wic":
        return 6
    elif task == "stsb":
        return 7
    elif task == "boolq":
        return 8
    elif task == "sst2":
        return 9
    elif task == "qnli":
        return 10
    elif task == "qqp":
        return 11
    elif task == "mnli":
        return 12
    elif task == "AVG":
        return 13
    else:
        return 999


VERSIONS = [
    "GSG/PPROBE/C-V5",
    # "GSG/PROPPROBEBE/C-V0",
    "GSG/PPROBE/st-a",
]
# This code is used to store the results of the various models in a dictionary
# so that they can be easily accessed later.
# The keys of the dictionary are the names of the models and the values are
# the results of the models.

# make directory v3, v3/differences, v3/plots, v3/differences/plots
# overwrite if exists
# shutil.rmtree("results/probe", ignore_errors=False)
os.makedirs("GSG/PPROBE/plots", exist_ok=True)
os.makedirs("GSG/PPROBE/differences/plots", exist_ok=True)

result_dict = {}
for i, VERSION in enumerate(VERSIONS):
    for OMEGA in ["00", "01", "03", "05", "07", "09", "1"]:
        if VERSION is not None:
            if not "st-a" in VERSION:
                actual_version = int(VERSION[-1])
                result_dict[f"C-V{actual_version} OMEGA-{OMEGA}"] = pd.read_csv(
                    f"{VERSION}_{OMEGA}.csv"
                )
                result_dict[f"C-V{actual_version} OMEGA-{OMEGA}"]["omega"] = OMEGA
                result_dict[f"C-V{actual_version} OMEGA-{OMEGA}"][
                    "version"
                ] = actual_version
            else:
                result_dict[f"st-a OMEGA-{OMEGA}"] = pd.read_csv(
                    f"{VERSION}_{OMEGA}.csv"
                )
                result_dict[f"st-a OMEGA-{OMEGA}"]["omega"] = OMEGA
                result_dict[f"st-a OMEGA-{OMEGA}"]["version"] = "st-a"


# for setup in result_dict.keys():
#     result_dict[setup] = result_dict[setup].sort_values(
#         by=["task", "train_pct"], ascending=[True, True]
#     )
for setup in result_dict.keys():
    # result_dict[setup] = result_dict[setup].sort_values(by=["task", "train_pct"], ascending=[False, True])
    result_dict[setup] = result_dict[setup].sort_values(
        by=["target_task", "train_pct"],
        key=lambda x: x.map(sort_tasks),
        ascending=[True, True],
    )


result_dict_drop = {
    k: v.drop(columns=["best_seed", "seeds", "omega"]).set_index(
        ["target_task", "train_pct"]
    )
    for k, v in result_dict.items()
}


# add new column to full, st_a, difference, based on
def compute_main_metric(task):
    if task == "cola":
        metric = "matthews_correlation"
    elif task == "stsb":
        metric = "pearson"
    else:
        metric = "accuracy"
    return metric


versions = {
    "C-V5": [],
    # "C-V0" : [],
    "st-a": [],
}
for df in result_dict.keys():
    # map omega to group
    # map version to group
    for base_version in versions.keys():
        if base_version in df:
            versions[base_version].append(df)

for base_version in versions.keys():
    print(base_version)
    if len(versions[base_version]) == 0:
        continue

    # loop over all df in version

    for i, version in enumerate(versions[base_version]):
        print(base_version, version)
        # add empty row for mrpc, qqp
        for target_task in []:
            for train_pct in [10, 25, 50, 100]:
                result_dict[version] = result_dict[version].append(
                    {
                        "target_task": target_task,
                        "source_task": None,
                        "train_pct": train_pct,
                        "accuracy_MEAN": 0,
                        "accuracy_STD": 0,
                        "accuracy_mm_MEAN": 0,
                        "accuracy_mm_STD": 0,
                        "pearson_MEAN": 0,
                        "pearson_STD": 0,
                        "spearmanr_MEAN": 0,
                        "spearmanr_STD": 0,
                        "matthews_correlation_MEAN": 0,
                        "matthews_correlation_STD": 0,
                        "omega": result_dict[version]["omega"][0],
                    },
                    ignore_index=True,
                )

        df = result_dict[version].copy()
        df["metric"] = df["target_task"].apply(compute_main_metric)
        df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
        df["metric_STD"] = df.apply(lambda x: x[x["metric"] + "_STD"], axis=1)
        df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))
        avg_target = df.groupby(["train_pct", "target_task"]).mean()["metric_MEAN"]
        avg_source = df.groupby(["train_pct", "source_task"]).mean()["metric_MEAN"]
        std_target = df.groupby(["train_pct", "target_task"]).mean()["metric_STD"]
        std_source = df.groupby(["train_pct", "source_task"]).mean()["metric_STD"]
        for train_pct in [100]:
            # loop over both avg and add to df
            for task in avg_target.index.get_level_values(1).unique():
                df = df.append(
                    {
                        "target_task": task,
                        "source_task": "AVG",
                        "train_pct": train_pct,
                        "metric_MEAN": avg_target[train_pct][task],
                        "metric_STD": std_target[train_pct][task],
                        "% of Training Data": str(train_pct),
                        "omega": df["omega"][0],
                    },
                    ignore_index=True,
                )

            for task in avg_source.index.get_level_values(1).unique():
                df = df.append(
                    {
                        "target_task": "AVG",
                        "source_task": task,
                        "train_pct": train_pct,
                        "metric_MEAN": avg_source[train_pct][task],
                        "metric_STD": std_source[train_pct][task],
                        "% of Training Data": str(train_pct),
                        "omega": df["omega"][0],
                    },
                    ignore_index=True,
                )
        if i == 0:
            base_df = df.copy()
        else:
            # append df to base_df
            base_df = base_df.append(df)
        base_df = (
            base_df.groupby(["target_task", "source_task", "train_pct", "omega"])
            .mean()
            .reset_index()
        )
        # and now 1 plot with all train_pcts: 2 rows, 2 columns
        # but for each task individually
    os.makedirs(
        f"GSG/PPROBE/plots/omegas/{base_version}/100",
        exist_ok=True,
    )
    for train_pct in ["100"]:
        filtered_df = base_df[base_df["train_pct"] == int(train_pct)]
        # map omega: 00 -> 0.0, 01 -> 0.1, etc.
        filtered_df["omega"] = filtered_df["omega"].apply(
            lambda x: float("0." + x[1]) if x != "1" else 1.0
        )
        filtered_df = filtered_df.sort_values(
            by=["target_task", "source_task", "train_pct"],
            key=lambda x: x.map(sort_tasks),
            ascending=[True, True, True],
        )

        # and now a plot with all tasks: 4 rows, 4 columns

        fig, ax = plt.subplots(
            len(filtered_df["target_task"].unique()),
            len(filtered_df["source_task"].unique()),
            figsize=(50, (len(filtered_df["target_task"].unique()) * 5)),
        )
        for row, target_task in enumerate(filtered_df["target_task"].unique()):
            for col, source_task in enumerate(filtered_df["source_task"].unique()):
                ax[row, col].set_box_aspect(1)
                df = filtered_df[
                    (filtered_df["target_task"] == target_task)
                    & (filtered_df["source_task"] == source_task)
                ]
                if source_task == "stsb":
                    # continue
                    pass
                s = sns.lineplot(
                    data=df,
                    x="omega",
                    y="metric_MEAN",
                    ax=ax[row, col],
                    markers=True,
                    marker="o",
                    markersize=6,  # set marker size to 10
                    markerfacecolor="blue",  # set marker color to blue
                    markeredgecolor="white",
                    color="red",
                    linestyle="--",  # set line style to "--"
                )
                # add error bars from metric_STD feature
                ax[row, col].errorbar(
                    df["omega"],
                    df["metric_MEAN"],
                    yerr=df["metric_STD"],
                    fmt="none",
                    ecolor="gray",
                    capsize=3,
                )
                s.set(ylim=(0.5, 1))
                # y axis ticks in 0.05 steps
                # 0.5 to 1.0 in 0.05 steps
                # for ax:
                ax[row, col].set_yticks(
                    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                )
                ax[row, col].set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                # add grid
                ax[row, col].grid(True)
                ax[row, col].set_title(f"{source_task}", fontsize=15)
                if col == 0:
                    ax[row, col].set_ylabel(f"{target_task}", fontsize=15)
                else:
                    ax[row, col].set_ylabel("")
                ax[row, col].set_xlabel("$\omega$")
                

        plt.suptitle(f"{base_version.upper()} - pairwise probing", fontsize=20)
        # entire x axis label
        fig.text(
            0.5,
            0.04,
            "Source Task",
            ha="center",
            fontsize=25,
        )
        # entire y axis label
        fig.text(
            0.02,
            0.5,
            "Target Task",
            va="center",
            rotation="vertical",
            fontsize=25,
            # make margins smaller
        )
        # tight_layout only for fig text
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

        plt.savefig(
            f"GSG/PPROBE/plots/omegas/{base_version}/{train_pct}/{base_version}_ALL_{train_pct}_pairwise.png",
            dpi=300,
        )
        plt.close()


print("done")
