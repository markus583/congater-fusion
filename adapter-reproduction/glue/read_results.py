import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

full = pd.read_csv("results/FULL.csv")
st_a = pd.read_csv("results/ST-A.csv")

# sort: first by column task, then by column train_pct
full = full.sort_values(by=["task", "train_pct"], ascending=[True, True])
st_a = st_a.sort_values(by=["task", "train_pct"], ascending=[True, True])

# compute difference, all matched except n_runs, best_seed, seeds
full_drop = full.drop(columns=["n_runs", "best_seed", "seeds"]).set_index(
    ["task", "train_pct"]
)
st_a_drop = st_a.drop(columns=["n_runs", "best_seed", "seeds"]).set_index(
    ["task", "train_pct"]
)

# positive means full is better, negative means st-a is better
difference = full_drop - st_a_drop
# multiindex to columns
difference = difference.reset_index()


# add new column to full, st_a, difference, based on
def compute_main_metric(task):
    if task == "cola":
        metric = "matthews_correlation"
    elif task == "stsb":
        metric = "pearson"
    else:
        metric = "accuracy"
    return metric


for df, name in zip(
    [full, st_a, difference],
    ["Full", "ST-A", "Difference (positive means full is better)"],
):
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)

    # plot
    s = sns.catplot(
        data=df,
        kind="bar",
        x="task",
        y="metric_MEAN",
        hue="train_pct",
        palette="Blues_d",
    )
    s.set_axis_labels("Task", "Metric Values")
    s.set_xticklabels(rotation=45)
    if name in ["Full", "ST-A"]:
        s.set(ylim=(0, 1))
        # y axis ticks in 0.05 steps
        plt.yticks([i / 20 for i in range(21)])
    # title
    s.fig.suptitle(f"{name} results")
    s.fig.set_size_inches((10, 6))
    s.savefig(f"results/{name}.png")

    df.to_csv(f"results/{name}_proc.csv", index=False)

print("done")
