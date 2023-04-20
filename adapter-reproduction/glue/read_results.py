from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

full = pd.read_csv("results/FULL.csv")
st_a = pd.read_csv("results/ST-A.csv")
st_a_fusion = pd.read_csv("results/ST-A_FUSION.csv")
ct_a_bert = pd.read_csv("results/CT-A_bert-init.csv")
ct_a_custom = pd.read_csv("results/CT-A_custom-init.csv")
ct_a_bert_LN = pd.read_csv("results/CT-A_bert-init-LN.csv")
ct_a_bert_RELU = pd.read_csv("results/CT-A_bert-init-RELU.csv")
ct_a_bert_rf4 = pd.read_csv("results/CT-A_bert-init-RF4.csv")
ct_a_bert_relu_plus = pd.read_csv("results/CT-A_bert-init-RELU-PLUS.csv")

result_dict = {
    "Full": full,
    "ST-A": st_a,
    "ST-A Fusion": st_a_fusion,
    "CT-A_Bert-init": ct_a_bert,
    "CT-A Bert-init RELU": ct_a_bert_RELU,
    "CT-A_Bert-init-LN": ct_a_bert_LN,
    "CT-A_Custom-init": ct_a_custom,
    "CT-A_Bert-init-rf4": ct_a_bert_rf4,
    "CT-A_Bert-init RELU ADDITIVE": ct_a_bert_relu_plus,
}

for setup in result_dict.keys():
    result_dict[setup] = result_dict[setup].sort_values(by=["task", "train_pct"], ascending=[True, True])


result_dict_drop = {k: v.drop(columns=["n_runs", "best_seed", "seeds"]).set_index(["task", "train_pct"]) for k, v in result_dict.items()}

# now the same with differences as dict
differences = {}

for setup in combinations(result_dict.keys(), 2):
    differences[f"{setup[0]}_VS_{setup[1]}"] = result_dict_drop[setup[0]] - result_dict_drop[setup[1]]
    differences[f"{setup[0]}_VS_{setup[1]}"] = differences[f"{setup[0]}_VS_{setup[1]}"].reset_index()


# add new column to full, st_a, difference, based on
def compute_main_metric(task):
    if task == "cola":
        metric = "matthews_correlation"
    elif task == "stsb":
        metric = "pearson"
    else:
        metric = "accuracy"
    return metric


for name, df in result_dict.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

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
    s.fig.suptitle(f"{name} results")
    # add grid
    s.ax.grid(True)

    s.fig.set_size_inches((10, 6))
    plt.tight_layout()
    s.savefig(f"results/plots/{name.split(' (')[0]}.png")

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/{name.split(' (')[0]}_proc.csv", index=False)


for name, df in differences.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

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
    s.fig.suptitle(f"{name.replace('_', ' ')} (higher means first is better)")
    s.ax.grid(True)

    s.fig.set_size_inches((10, 6))
    plt.tight_layout()
    s.savefig(f"results/differences/plots/{name}.png")

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/differences/{name}.csv", index=False)
print("done")
