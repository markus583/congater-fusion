from itertools import combinations
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.chdir(os.path.dirname(os.path.realpath(__file__)))

full = pd.read_csv("results/FULL.csv")
st_a = pd.read_csv("results/ST-A.csv")
st_a_fusion = pd.read_csv("results/ST-A_FUSION.csv")
ct_a_bert = pd.read_csv("results/CT-A_bert-init.csv")
ct_a_bert_RELU = pd.read_csv("results/CT-A_bert-init-RELU.csv")
ct_a_bert_RELU_ln_before = pd.read_csv("results/CT-A_bert-init-RELU-LN_BEFORE.csv")
ct_a_bert_relu_plus = pd.read_csv("results/CT-A_bert-init-RELU-PLUS.csv")
ct_a_bert_relu_plus_ln_before = pd.read_csv("results/CT-A_bert-init-RELU-PLUS-LN_BEFORE.csv")
ct_a_bert_relu_gateadp = pd.read_csv("results/CT-A_bert-init-RELU-gate_adapter.csv")
ct_a_bert_relu_gateadp_ln_after = pd.read_csv("results/CT-A_bert-init-RELU-LN_AFTER-gate_adapter.csv")
ct_a_bert_relu_gateadp_ln_before = pd.read_csv("results/CT-A_bert-init-RELU-LN_BEFORE-gate_adapter.csv")
ct_a_bert_relu_gateadp_ln_ab = pd.read_csv("results/CT-A_bert-init-RELU-LN_AB-gate_adapter.csv")
ct_a_bert_relu_plus_gateadp = pd.read_csv("results/CT-A_bert-init-RELU-PLUS-gate_adapter.csv")
ct_a_bert_relu_plus_gateadp_ln_before = pd.read_csv("results/CT-A_bert-init-RELU-PLUS-LN_BEFORE-gate_adapter.csv")
ct_a_bert_relu_plus_gateadp_ln_after = pd.read_csv("results/CT-A_bert-init-RELU-PLUS-LN_AFTER-gate_adapter.csv")

# This code is used to store the results of the various models in a dictionary
# so that they can be easily accessed later.
# The keys of the dictionary are the names of the models and the values are
# the results of the models.
result_dict = {
    "Full": full,
    "ST-A": st_a,
    "ST-A Fusion": st_a_fusion,
    "CT-A_Bert-init": ct_a_bert,
    "CT-A Bert-init RELU": ct_a_bert_RELU,
    "CT-A Bert-init RELU, LN BEFORE x": ct_a_bert_RELU_ln_before,
    "CT-A Bert-init RELU ADDITIVE": ct_a_bert_relu_plus,
    "CT-A Bert-init RELU ADDITIVE, LN BEFORE +": ct_a_bert_relu_plus_ln_before,
    "CT-A Bert-init RELU GATE ADAPTER": ct_a_bert_relu_gateadp,
    "CT-A Bert-init RELU GATE ADAPTER, LN AFTER x": ct_a_bert_relu_gateadp_ln_after,
    "CT-A Bert-init RELU GATE ADAPTER, LN BEFORE ADP": ct_a_bert_relu_gateadp_ln_before,
    "CT-A Bert-init RELU GATE ADAPTER, LN BEFORE ADP AFTER X": ct_a_bert_relu_gateadp_ln_ab,
    "CT-A Bert-init RELU ADDITIVE GATE ADAPTER": ct_a_bert_relu_plus_gateadp,
    "CT-A Bert-init RELU ADDITIVE GATE ADAPTER, LN BEFORE +": ct_a_bert_relu_plus_gateadp_ln_before,
    "CT-A Bert-init RELU ADDITIVE GATE ADAPTER, LN AFTER +": ct_a_bert_relu_plus_gateadp_ln_after,
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
    
    
    n_complete_runs = len(df[df['train_pct'] == 100])
    if n_complete_runs == 8:
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
    s.fig.suptitle(f"{name} results")
    # add grid
    s.ax.grid(True)

    s.fig.set_size_inches((10, 6))
    plt.tight_layout()
    s.savefig(f"results/v2/plots/{name.split(' (')[0]}.png")
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/v2/{name.split(' (')[0]}_proc.csv", index=False)


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
    s.savefig(f"results/v2/differences/plots/{name}.png")
    # close
    plt.close()

    # df.to_csv(f"results/{name}_proc.csv", index=False)
    # same but only take content before ( of name
    df.to_csv(f"results/v2/differences/{name}.csv", index=False)
print("done")
