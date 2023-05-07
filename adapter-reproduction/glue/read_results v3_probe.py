from itertools import combinations
import os
import warnings
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))


VERSIONS = [
    None,
    "results/probe/ct_0-a-RELUcopy",
    "results/probe/ct_2-a-RELU-PLUS-gate_adapter",
    "results/probe/ct_3-a-V3",
    "results/probe/ct_3-a-V4",
    "results/probe/ct_2-a-RELU-PLUS-LN_BEFORE-gate_adapter"
]
# TODO: add V0, (local, still running)

# This code is used to store the results of the various models in a dictionary
# so that they can be easily accessed later.
# The keys of the dictionary are the names of the models and the values are
# the results of the models.

# make directory v3, v3/differences, v3/plots, v3/differences/plots
# overwrite if exists
# shutil.rmtree("results/probe", ignore_errors=False)
os.makedirs("results/probe/plots", exist_ok=True)
os.makedirs("results/probe/differences/plots", exist_ok=True)

result_dict = {}
for i, VERSION in enumerate(VERSIONS):
    for OMEGA in ["00", "01", "03", "05", "07", "09", "1"]:
        if VERSION is not None:
            result_dict[f"V-{i} OMEGA-{OMEGA}"] = pd.read_csv(f"{VERSION}_{OMEGA}.csv")
            result_dict[f"V-{i} OMEGA-{OMEGA}"]["omega"] = OMEGA
            result_dict[f"V-{i} OMEGA-{OMEGA}"]["version"] = i


for setup in result_dict.keys():
    result_dict[setup] = result_dict[setup].sort_values(
        by=["task", "train_pct"], ascending=[True, True]
    )


result_dict_drop = {
    k: v.drop(columns=["best_seed", "seeds", "omega"]).set_index(["task", "train_pct"])
    for k, v in result_dict.items()
}

# this is a message
# now the same with differences as dict
differences = {}

for setup in combinations(result_dict.keys(), 2):
    differences[f"{setup[0]}_VS_{setup[1]}"] = (
        result_dict_drop[setup[0]] - result_dict_drop[setup[1]]
    )
    differences[f"{setup[0]}_VS_{setup[1]}"] = differences[
        f"{setup[0]}_VS_{setup[1]}"
    ].reset_index()
    # add new column to differences, based on 1) n_runs of first, 2) n_runs of second
    differences[f"{setup[0]}_VS_{setup[1]}"]["n_runs_0"] = float(
        str(result_dict_drop[setup[0]]["n_runs"].mean())[:3]
    )
    differences[f"{setup[0]}_VS_{setup[1]}"]["n_runs_1"] = float(
        str(result_dict_drop[setup[1]]["n_runs"].mean())[:3]
    )


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
    "V-0": [],
    "V-1": [],
    "V-2": [],
    "V-3": [],
    "V-4": [],
    "V-5": []
}
for df in result_dict.keys():
    # map omega to group
    # map version to group
    versions["V-" + str(result_dict[df]["version"].iloc[0])].append(df)

for base_version in versions.keys():
    print(base_version)
    if len(versions[base_version]) == 0:
        continue
    # loop over all df in version

    for i, version in enumerate(versions[base_version]):
        print(base_version, version)
        df = result_dict[version].copy()
        df["metric"] = df["task"].apply(compute_main_metric)
        df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
        df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))
        avg = df.groupby("train_pct").mean()["metric_MEAN"]
        for train_pct in [10, 25, 50, 100]:
            df = df.append(
                {
                    "task": "AVG",
                    "train_pct": train_pct,
                    "metric_MEAN": avg[train_pct],
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
        base_df = base_df.groupby(["task", "train_pct", "omega"]).mean().reset_index()
        # and now 1 plot with all train_pcts: 2 rows, 2 columns
        # but for each task individually
    os.makedirs(
        f"results/probe/plots/omegas/{base_version}/ALL",
        exist_ok=True,
    )
    for task in base_df["task"].unique():
        # filtered_df = base_df[base_df["train_pct"] == int(train_pct)]
        # map omega: 00 -> 0.0, 01 -> 0.1, etc.
        filtered_df = base_df.copy()
        filtered_df["omega"] = base_df["omega"].apply(
            lambda x: float("0." + x[1]) if x != "1" else 1.0
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        # first ax: 10%, second ax: 25%, third ax: 50%, fourth ax: 100%
        for i, train_pct in enumerate(["10", "25", "50", "100"]):
            sns.lineplot(
                data=filtered_df[
                    (filtered_df["task"] == task)
                    & (filtered_df["train_pct"] == int(train_pct))
                ],
                x="omega",
                y="metric_MEAN",
                ax=axes[i // 2, i % 2],
                markers=True,
                marker="o",
                markersize=8,                # set marker size to 10
                markerfacecolor="blue",       # set marker color to blue
                markeredgecolor="white",
                color="red",
                linestyle="--"               # set line style to "--"
            )
            # ylim
            axes[i // 2, i % 2].set(ylim=(0, 1))
            # y axis ticks in 0.05 steps
            axes[i // 2, i % 2].set_yticks([i / 20 for i in range(21)])
            axes[i // 2, i % 2].set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
            # add grid
            axes[i // 2, i % 2].grid(True)
            axes[i // 2, i % 2].set_title(
                f"Task {task} - {train_pct}% of Training Data"
            )
            axes[i // 2, i % 2].set_ylabel("Metric")
            axes[i // 2, i % 2].set_xlabel("Omega")
            # add title
            plt.suptitle(f"Task {task} - {base_version}")
            if base_version == "V-5":
                plt.suptitle(f"Task {task} - V2 LN-Before x")
            
        plt.tight_layout()
        plt.savefig(
            f"results/probe/plots/omegas/{base_version}/ALL/{base_version}_{task}.png",
            dpi=300,
        )
        plt.close()
    for train_pct in ["10", "25", "50", "100"]:
        filtered_df = base_df[base_df["train_pct"] == int(train_pct)]
        # map omega: 00 -> 0.0, 01 -> 0.1, etc.
        filtered_df["omega"] = filtered_df["omega"].apply(
            lambda x: float("0." + x[1]) if x != "1" else 1.0
        )
        for task in filtered_df["task"].unique():
            os.makedirs(
                f"results/probe/plots/omegas/{base_version}/{train_pct}", exist_ok=True
            )
            # sns line plot
            # x = omega, y = metric_MEAN
            fig, ax = plt.subplots(figsize=(10, 5))
            s = sns.lineplot(
                data=filtered_df[filtered_df["task"] == task],
                x="omega",
                y="metric_MEAN",
                ax=ax,
                markers=True,
                marker="o",
                markersize=8,                # set marker size to 10
                markerfacecolor="blue",       # set marker color to blue
                markeredgecolor="white",
                color="red",
                linestyle="--"               # set line style to "--"
            )
            s.set(ylim=(0, 1))
            # y axis ticks in 0.05 steps
            plt.yticks([i / 20 for i in range(21)])
            ax.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
            # add grid
            ax.grid(True)
            ax.set_title(
                f"Task {task} - {train_pct}% of Training Data - {base_version}"
            )
            ax.set_ylabel("Metric")
            ax.set_xlabel("Omega")
            plt.savefig(
                f"results/probe/plots/omegas/{base_version}/{train_pct}/{base_version}_{task}_{train_pct}.png",
                dpi=300,
            )
            plt.close()

 
        


for name, df in result_dict.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

    # compute average over all tasks for different train percentages
    avg = df.groupby("train_pct").mean()["metric_MEAN"]
    # add 1 row for each train_pct
    for train_pct in [10, 25, 50, 100]:
        df = df.append(
            {
                "task": "AVG",
                "train_pct": train_pct,
                "metric_MEAN": avg[train_pct],
                "% of Training Data": str(train_pct),
            },
            ignore_index=True,
        )

    n_complete_runs = len(df[df["train_pct"] == 100])
    if n_complete_runs == 9:
        print(name)
        # average metric value with train_pct 10
        mean_10 = df[df["train_pct"] == 10]["metric_MEAN"].mean()
        std_10 = df[df["train_pct"] == 10]["metric_MEAN"].std()
        print(
            "Mean + std metric value with train_pct 10: "
            + str(mean_10)
            + " +- "
            + str(std_10)
        )
        # average metric value with train_pct 25
        mean_25 = df[df["train_pct"] == 25]["metric_MEAN"].mean()
        std_25 = df[df["train_pct"] == 25]["metric_MEAN"].std()
        print(
            "Mean + std metric value with train_pct 25: "
            + str(mean_25)
            + " +- "
            + str(std_25)
        )
        # average metric value with train_pct 50
        mean_50 = df[df["train_pct"] == 50]["metric_MEAN"].mean()
        std_50 = df[df["train_pct"] == 50]["metric_MEAN"].std()
        print(
            "Mean + std metric value with train_pct 50: "
            + str(mean_50)
            + " +- "
            + str(std_50)
        )
        # average metric value with train_pct 100
        mean_100 = df[df["train_pct"] == 100]["metric_MEAN"].mean()
        std_100 = df[df["train_pct"] == 100]["metric_MEAN"].std()
        print(
            "Mean + std metric value with train_pct 100: "
            + str(mean_100)
            + " +- "
            + str(std_100)
        )
        print("----------------------")

    # plot
    s = sns.catplot(
        data=df,
        kind="bar",
        x="task",
        y="metric_MEAN",
        hue="% of Training Data",
        palette="Blues_d",
    )
    s.set_axis_labels("Task", "Metric Values")
    s.set_xticklabels(rotation=45)
    s.set(ylim=(0, 1))
    # y axis ticks in 0.05 steps
    plt.yticks([i / 20 for i in range(21)])
    # title
    s.fig.suptitle(f"{name} results ({df['n_runs'][0]})")
    # add grid
    s.ax.grid(True)

    s.fig.set_size_inches((10, 6))
    plt.tight_layout()
    s.savefig(f"results/probe/plots/{name.split(' (')[0]}.png")
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/probe/{name.split(' (')[0]}_proc.csv", index=False)


for name, df in differences.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

    # compute average over all tasks for different train percentages
    avg = df.groupby("train_pct").mean()["metric_MEAN"]
    # add 1 row for each train_pct
    for train_pct in [10, 25, 50, 100]:
        df = df.append(
            {
                "task": "AVG",
                "train_pct": train_pct,
                "metric_MEAN": avg[train_pct],
                "% of Training Data": str(train_pct),
            },
            ignore_index=True,
        )

    # plot
    s = sns.catplot(
        data=df,
        kind="bar",
        x="task",
        y="metric_MEAN",
        hue="% of Training Data",
        palette="Blues_d",
    )
    s.set_axis_labels("Task", "Metric Values")
    s.set_xticklabels(rotation=45)
    # title
    s.fig.suptitle(
        f"{name.replace('_', ' ')} (higher means first is better) ({df['n_runs_0'][0], df['n_runs_1'][0]} seeds)"
    )
    s.ax.grid(True)
    # y axis ticks in 0.01 steps
    # but we need min and max of differences for scaling
    max = df["metric_MEAN"].max()
    min = df["metric_MEAN"].min()
    if min < -0.05 or max > 0.1:
        plt.yticks([i / 100 for i in range(int(min * 100), int(max * 100) + 1)])
    else:
        # if min and max are smaller than 0.1, we can use 0.01 steps from -0.1 to 0.1
        plt.yticks([i / 100 for i in range(-5, 11)])

    s.fig.set_size_inches((12, 6))
    # plt.tight_layout()
    s.savefig(f"results/probe/differences/plots/{name}.png")
    # close
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/probe/differences/{name}.csv", index=False)

print("done")
