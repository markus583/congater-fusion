from itertools import combinations
import os
import warnings
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

full = pd.read_csv("SuperGLUE/results/FULL.csv")
st_a = pd.read_csv("SuperGLUE/results/ST-A.csv")
c_v2 = pd.read_csv("SuperGLUE/results/C-V2.csv")
st_a_fusion = pd.read_csv("SuperGLUE/results/ST-A-FUSION-SUPERGLUE.csv")
c_v2_fusion = pd.read_csv("SuperGLUE/results/C-V2-FUSION-SUPERGLUE.csv")

# This code is used to store the results of the various models in a dictionary
# so that they can be easily accessed later.
# The keys of the dictionary are the names of the models and the values are
# the results of the models.

# make directory v3, v3/differences, v3/plots, v3/differences/plots
# overwrite if exists
# shutil.rmtree("results/v3", ignore_errors=False)
# os.makedirs("results/v3/plots", exist_ok=True)
# os.makedirs("results/v3/differences/plots", exist_ok=True)
    
result_dict = {
    "Full": full,
    "ST-A": st_a,    
    "C-V2": c_v2,
    "ST-A Fusion": st_a_fusion,
    "C-V2 Fusion": c_v2_fusion,
}


for setup in result_dict.keys():
    result_dict[setup] = result_dict[setup].sort_values(by=["task", "train_pct"], ascending=[True, True])


result_dict_drop = {k: v.drop(columns=["best_seed", "seeds"]).set_index(["task", "train_pct"]) for k, v in result_dict.items()}

# this is a message
# now the same with differences as dict
differences = {}

for setup in combinations(result_dict.keys(), 2):
    differences[f"{setup[0]}_VS_{setup[1]}"] = result_dict_drop[setup[0]] - result_dict_drop[setup[1]]
    differences[f"{setup[0]}_VS_{setup[1]}"] = differences[f"{setup[0]}_VS_{setup[1]}"].reset_index()
    # add new column to differences, based on 1) n_runs of first, 2) n_runs of second
    differences[f"{setup[0]}_VS_{setup[1]}"]["n_runs_0"] = float(str(result_dict_drop[setup[0]]["n_runs"].mean())[:3])
    differences[f"{setup[0]}_VS_{setup[1]}"]["n_runs_1"] = float(str(result_dict_drop[setup[1]]["n_runs"].mean())[:3])


# add new column to full, st_a, difference, based on
def compute_main_metric(task):
    if task in ["multirc", "record"]:
        metric = "f1"
    else:
        metric = "accuracy"
    return metric


for name, df in result_dict.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))
    
    # compute average over all tasks for different train percentages
    avg = df.groupby("train_pct").mean()["metric_MEAN"]
    # add 1 row for each train_pct
    for train_pct in [10, 25, 50, 100]:
        df = df.append({"task": "AVG", "train_pct": train_pct, "metric_MEAN": avg[train_pct], "% of Training Data": str(train_pct)}, ignore_index=True)
    
    
    n_complete_runs = len(df[df['train_pct'] == 100])
    if n_complete_runs == 7:
        print(name)
        # average metric value with train_pct 10
        mean_10 = df[df["train_pct"] == 10]["metric_MEAN"].mean()
        std_10 = df[df["train_pct"] == 10]["metric_MEAN"].std()
        print("Mean + std metric value with train_pct 10: " + str(mean_10) + " +- " + str(std_10))
        # average metric value with train_pct 25
        mean_25 = df[df["train_pct"] == 25]["metric_MEAN"].mean()
        std_25 = df[df["train_pct"] == 25]["metric_MEAN"].std()
        print("Mean + std metric value with train_pct 25: " + str(mean_25) + " +- " + str(std_25))
        # average metric value with train_pct 50
        mean_50 = df[df["train_pct"] == 50]["metric_MEAN"].mean()
        std_50 = df[df["train_pct"] == 50]["metric_MEAN"].std()
        print("Mean + std metric value with train_pct 50: " + str(mean_50) + " +- " + str(std_50))
        # average metric value with train_pct 100
        mean_100 = df[df["train_pct"] == 100]["metric_MEAN"].mean()
        std_100 = df[df["train_pct"] == 100]["metric_MEAN"].std()
        print("Mean + std metric value with train_pct 100: " + str(mean_100) + " +- " + str(std_100))
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
    s.fig.suptitle(f"{name} results ({str(df['n_runs'].mean())[:3]})")
    # add grid
    s.ax.grid(True)

    s.fig.set_size_inches((10, 6))
    # plt.tight_layout()
    s.savefig(f"SuperGLUE/results/plots/{name.split(' (')[0]}.png")
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"SuperGLUE/results/csv/{name.split(' (')[0]}_proc.csv", index=False)


for name, df in differences.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

    # compute average over all tasks for different train percentages
    avg = df.groupby("train_pct").mean()["metric_MEAN"]
    # add 1 row for each train_pct
    for train_pct in [10, 25, 50, 100]:
        df = df.append({"task": "AVG", "train_pct": train_pct, "metric_MEAN": avg[train_pct], "% of Training Data": str(train_pct)}, ignore_index=True)

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
    s.fig.suptitle(f"{name.replace('_', ' ')} (higher means first is better) ({df['n_runs_0'][0], df['n_runs_1'][0]} seeds)")
    s.ax.grid(True)
    # y axis ticks in 0.01 steps
    # but we need min and max of differences for scaling
    max = df["metric_MEAN"].max()
    min = df["metric_MEAN"].min()
    if min < - 0.05 or max > 0.1:
        plt.yticks([i / 100 for i in range(int(min * 100), int(max * 100) + 1)])
    else:
        # if min and max are smaller than 0.1, we can use 0.01 steps from -0.1 to 0.1
        plt.yticks([i / 100 for i in range(-5, 11)])

    s.fig.set_size_inches((12, 6))
    # plt.tight_layout()
    s.savefig(f"SuperGLUE/results/plots/differences/plots/{name}.png")
    # close
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"SuperGLUE/results/csv/differences/{name}.csv", index=False)
print("done")
