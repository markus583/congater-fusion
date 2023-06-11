from itertools import combinations
import os
import warnings
import shutil

from probe_grid import probing

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
    # "GSG/PPROBE/C-V5",
    # "GSG/PROPPROBEBE/C-V0",
    "GSG/PROBE_GRID/st-a",
]

SOURCE_TASK_PARIS = [
    ["omega_mnli", "omega_qqp"],
    ["omega_mnli", "omega_SELF",],
]
# This code is used to store the results of the various models in a dictionary
# so that they can be easily accessed later.
# The keys of the dictionary are the names of the models and the values are
# the results of the models.

# make directory v3, v3/differences, v3/plots, v3/differences/plots
# overwrite if exists
# shutil.rmtree("results/probe", ignore_errors=False)
os.makedirs("GSG/PROBE_GRID/plots", exist_ok=True)
# os.makedirs("GSG/PROBE_GRID/differences/plots", exist_ok=True)

result_dict = {}
for i, VERSION in enumerate(VERSIONS):
    if VERSION is not None:
        # get unique filenames
        all_files = os.listdir(VERSION[:-4])
        for file in all_files:
            if file.endswith(".csv"):
                result_dict[file] = pd.read_csv(f"{VERSION[:-4]}/{file}")

for setup in result_dict.keys():
    # result_dict[setup] = result_dict[setup].sort_values(by=["task", "train_pct"], ascending=[False, True])
    result_dict[setup] = result_dict[setup].sort_values(
        by=["target_task", "train_pct"],
        key=lambda x: x.map(sort_tasks),
        ascending=[True, True],
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
    # "C-V5": [],
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

        base_df = result_dict[version].copy()
        base_df["metric"] = base_df["target_task"].apply(compute_main_metric)
        base_df["metric_MEAN"] = base_df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
        base_df["metric_STD"] = base_df.apply(lambda x: x[x["metric"] + "_STD"], axis=1)
        base_df["% of Training Data"] = base_df["train_pct"].apply(lambda x: str(x))
        for source_tasks in SOURCE_TASK_PARIS:
            print(source_tasks)
            source_tasks_stripped = [task.split("_")[1] for task in source_tasks]
            df_source = base_df.dropna(subset=source_tasks)
            avg_task = df_source.groupby(["train_pct", source_tasks[0], source_tasks[1]]).mean()[
                "metric_MEAN"
            ]
            std_target = df_source.groupby(["train_pct", source_tasks[0], source_tasks[1]]).std()[
                "metric_MEAN"
            ]
            for train_pct in [100]:
                # loop over both avg and add to df
                for omega_1 in df_source[source_tasks[0]].unique():
                    for omega_2 in df_source[source_tasks[1]].unique():
                        df_source = df_source.append(
                            {
                                "target_task": "AVG",
                                "train_pct": train_pct,
                                "metric_MEAN": avg_task[train_pct][omega_1][omega_2],
                                "metric_STD": std_target[train_pct][omega_1][omega_2],
                                "% of Training Data": str(train_pct),
                                source_tasks[0]: omega_1,
                                source_tasks[1]: omega_2,
                            },
                            ignore_index=True,
                        )

            os.makedirs(
                f"GSG/PPROBE/plots/omegas/{base_version}/100",
                exist_ok=True,
            )
            for train_pct in ["100"]:
                filtered_df = df_source[df_source["train_pct"] == int(train_pct)]

                for target_task in filtered_df["target_task"].unique():
                    # get columns starting with omega_
                    fig, ax = plt.subplots(
                        2,
                        len(filtered_df[source_tasks[0]].unique()) + 1,
                        figsize=(30, 10),
                    )
                    df_task = filtered_df[filtered_df["target_task"] == target_task]
                    # drop rows with omega_mnli or omega_qqp nan

                    df_task = df_task.sort_values(
                        by=[source_tasks[0], source_tasks[1]],
                    )
                    # take mean and std of each omega:
                    # first, omega_mnli is ANY, omega_qqp is x_axis
                    # second, omega_mnli is x_axis, omega_qqp is ANY

                    for col, omega_value in enumerate(df_task[source_tasks[0]].unique()):
                        for row, source_task_2 in enumerate(range(len(source_tasks))):
                            # if last col
                            x_axis = source_tasks[1] if row == 0 else source_tasks[0]
                            df = df_task[df_task[source_tasks[row]] == omega_value]
                            # x axis: if row == 0, 1, if row == 1, 0

                            # lineplot for each source_task_2
                            s = sns.lineplot(
                                data=df,
                                x=x_axis,
                                y="metric_MEAN",
                                ax=ax[row, col],
                                markers=True,
                                marker="o",
                                dashes=False,
                                markerfacecolor="blue",  # set marker color to blue
                                markeredgecolor="white",
                                color="red",
                                linestyle="--",  # set line style to "--"
                            )
                            task_row = source_tasks[row].split("_")[1]
                            task_x_axis = x_axis.split("_")[1]
                            s.set_title(f"$\omega_{{{task_row}}} = {omega_value}$")
                            # add error bars from metric_STD feature
                            ax[row, col].errorbar(
                                df[x_axis],
                                df["metric_MEAN"],
                                yerr=df["metric_STD"],
                                fmt="none",
                                ecolor="gray",
                                capsize=3,
                            )
                            ax[row, col].set_box_aspect(1)
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
                            ax[row, col].set_xlabel(
                                "$\omega_{" + task_x_axis + "}$", fontsize=15
                            )
                            if col == 0:
                                ax[row, col].set_ylabel(f"Metric Values", fontsize=12)
                            else:
                                ax[row, col].set_ylabel("")
                    else:
                        for row, source_task_2 in enumerate(range(len(source_tasks))):
                            x_axis = source_tasks[1] if row == 0 else source_tasks[0]
                            df = df_task.groupby([x_axis]).mean().reset_index()
                            # lineplot for each source_task_2
                            s = sns.lineplot(
                                data=df,
                                x=x_axis,
                                y="metric_MEAN",
                                ax=ax[row, 7],
                                markers=True,
                                marker="o",
                                dashes=False,
                                markerfacecolor="blue",  # set marker color to blue
                                markeredgecolor="white",
                                color="red",
                                linestyle="--",  # set line style to "--"
                            )
                            task_row = source_tasks[row].split("_")[1]
                            task_x_axis = x_axis.split("_")[1]
                            s.set_title(f"$\omega_{{{task_row}}} = AVG$")
                            # add error bars from metric_STD feature
                            ax[row, 7].errorbar(
                                df[x_axis],
                                df["metric_MEAN"],
                                yerr=df["metric_STD"],
                                fmt="none",
                                ecolor="gray",
                                capsize=3,
                            )
                            ax[row, 7].set_box_aspect(1)
                            s.set(ylim=(0.5, 1))
                            # y axis ticks in 0.05 steps
                            # 0.5 to 1.0 in 0.05 steps
                            # for ax:
                            ax[row, 7].set_yticks(
                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                            )
                            ax[row, 7].set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                            # add grid
                            ax[row, 7].grid(True)
                            ax[row, 7].set_xlabel("$\omega_{" + task_x_axis + "}$", fontsize=15)
                            ax[row, 7].set_ylabel("")

                    plt.suptitle(
                        f"{base_version.upper()} - $\omega$ grid over {task_row} + {task_x_axis}\nTask {target_task}",
                        fontsize=20,
                    )
                    # entire x axis label
                    # tight_layout only for fig text
                    # fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

                    plt.savefig(
                        f"GSG/PROBE_GRID/plots/{base_version}_{source_tasks[0].split('_')[1]}-{source_tasks[1].split('_')[1]}_{target_task}.png",
                        dpi=300,
                    )
                    plt.close()
                    # TODO: AVG OVER TASKS
                    print(source_tasks)
                    print(f"10 best overall for {target_task}:")
                    best_10 = df_task.sort_values(by=["metric_MEAN"], ascending=False).head(10)
                    for i, (_, row) in enumerate(best_10.iterrows()):
                        print(
                            i,
                            round(row["metric_MEAN"], 3),
                            round(row["metric_STD"], 3),
                            row[source_tasks[0]],
                            row[source_tasks[1]],
                        )
                        

                    
                # sort by source_tasks
                filtered_df = filtered_df.sort_values(by=source_tasks)
                for i, source_task in enumerate(source_tasks_stripped):
                    fig, ax = plt.subplots(
                        len(filtered_df["target_task"].unique()),
                        len(filtered_df[source_tasks[0]].unique()) + 1,
                        figsize=(40, (len(filtered_df["target_task"].unique()) * 6)),
                    )
                    other = source_tasks_stripped[0] if i == 1 else source_tasks_stripped[1]
                    for row, task in enumerate(filtered_df["target_task"].unique()):
                        for col, omega_value in enumerate(filtered_df["omega_" + source_task].unique()):
                            # if last col
                            x_axis = "omega_" + other  # if row == 0 else source_tasks[0]
                            df = filtered_df[(filtered_df["omega_" + source_task] == omega_value) & (filtered_df["target_task"] == task)]
                            # x axis: if row == 0, 1, if row == 1, 0

                            # lineplot for each source_task_2
                            s = sns.lineplot(
                                data=df,
                                x=x_axis,
                                y="metric_MEAN",
                                ax=ax[row, col],
                                markers=True,
                                marker="o",
                                dashes=False,
                                markerfacecolor="blue",  # set marker color to blue
                                markeredgecolor="white",
                                color="red",
                                linestyle="--",  # set line style to "--"
                            )
                            s.set_title(f"$\omega_{{{source_task}}} = {omega_value}$", fontsize=20)
                            # add error bars from metric_STD feature
                            ax[row, col].errorbar(
                                df[x_axis],
                                df["metric_MEAN"],
                                yerr=df["metric_STD"],
                                fmt="none",
                                ecolor="gray",
                                capsize=3,
                            )
                            ax[row, col].set_box_aspect(1)
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
                            ax[row, col].set_xlabel(
                                "$\omega_{" + other + "}$", fontsize=20
                            )
                            if col == 0:
                                ax[row, col].set_ylabel(task, fontsize=20)
                            else:
                                ax[row, col].set_ylabel("")
                    else:
                        for row, target_task in enumerate(filtered_df["target_task"].unique()):
                            x_axis = "omega_" + other # if row == 0 else source_tasks[0]
                            df = filtered_df[(filtered_df["target_task"] == target_task)]
                            df = df.groupby([x_axis,]).mean().reset_index()
                            # lineplot for each source_task_2
                            s = sns.lineplot(
                                data=df,
                                x=x_axis,
                                y="metric_MEAN",
                                ax=ax[row, 7],
                                markers=True,
                                marker="o",
                                dashes=False,
                                markerfacecolor="blue",  # set marker color to blue
                                markeredgecolor="white",
                                color="red",
                                linestyle="--",  # set line style to "--"
                            )
                            s.set_title(f"$\omega_{{{source_task}}} = AVG$", fontsize=20)
                            # add error bars from metric_STD feature
                            ax[row, 7].errorbar(
                                df[x_axis],
                                df["metric_MEAN"],
                                yerr=df["metric_STD"],
                                fmt="none",
                                ecolor="gray",
                                capsize=3,
                            )
                            ax[row, 7].set_box_aspect(1)
                            s.set(ylim=(0.5, 1))
                            # y axis ticks in 0.05 steps
                            # 0.5 to 1.0 in 0.05 steps
                            # for ax:
                            ax[row, 7].set_yticks(
                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                            )
                            ax[row, 7].set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                            # add grid
                            ax[row, 7].grid(True)
                            ax[row, 7].set_xlabel("$\omega_{" + other + "}$", fontsize=20)
                            ax[row, 7].set_ylabel("")

                    plt.suptitle(
                        f"{base_version.upper()} - $\omega_{{{source_task}}}$ grid over {task_row} + {task_x_axis}\nAll tasks",
                        fontsize=20,
                    )
                    # entire x axis label
                    fig.text(
                        0.5,
                        0.04,
                        f"$\omega_{{{source_task}}}$",
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
                    # entire x axis label
                    # tight_layout only for fig text
                    # fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

                    plt.savefig(
                        f"GSG/PROBE_GRID/plots/{base_version}_{source_tasks[0].split('_')[1]}-{source_tasks[1].split('_')[1]}_ALL-{source_task}.png",
                        dpi=300,
                    )
                    plt.close()



print("done")
