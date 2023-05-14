from itertools import combinations
import os
import warnings
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

full = pd.read_csv("GSG/results/FULL.csv")
st_a = pd.read_csv("GSG/results/ST-A.csv")
c_v2 = pd.read_csv("GSG/results/C-V2.csv")
# st_a_fusion = pd.read_csv("GSG/results/ST-A-FUSION-GSG.csv")
# c_v2_fusion = pd.read_csv("GSG/results/C-V2-FUSION-GSG.csv")
st_a_fusion_fp16 = pd.read_csv("GSG/results/ST-A-FUSION-GSG-FP16.csv")
c_v2_fusion_fp16 = pd.read_csv("GSG/results/C-V2-FUSION-GSG-FP16.csv")

    
result_dict = {
    "Full": full,
    "ST-A": st_a,    
    "C-V2": c_v2,
    # "ST-A Fusion": st_a_fusion,
    # "C-V2 Fusion": c_v2_fusion,
    "ST-A Fusion": st_a_fusion_fp16,
    "C-V2 Fusion": c_v2_fusion_fp16,
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
    elif task == "cola":
        metric = "matthews_correlation"
    elif task == "stsb":
        metric = "pearson"
    else:
        metric = "accuracy"
    return metric


for name, df in differences.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

    # compute average over all tasks for different train percentages
    avg = df.groupby("train_pct").mean()["metric_MEAN"]
    # add 1 row for each train_pct
    for train_pct in [10, 25, 50, 100]:
        df = df.append({"task": "AVG", "train_pct": train_pct, "metric_MEAN": avg[train_pct], "% of Training Data": str(train_pct)}, ignore_index=True)

    df.to_csv(f"GSG/results/differences/csv/{name}.csv", index=False)
    
    px.bar(
        df,
        x="task",
        y="metric_MEAN",
        color="% of Training Data",
        # blues colormap, 4 options - light blue, medium blue, dark blue, very dark blue
        color_discrete_sequence=["#d1e5f0", "#92c5de", "#4393c3", "#2166ac"],
        barmode="group",
        title=f"{name.replace('_', ' ').replace('VS', 'vs.')} (higher means first is better) - {df['n_runs_0'][0], df['n_runs_1'][0]} seeds",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
    ).update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
        yaxis_range=[-0.1, 0.1],
        yaxis_tick0=-0.1,
        yaxis_dtick=0.01,
    ).write_image(
        f"GSG/results/differences/plots/all/{name}.png",
        scale=3
    )


for name, df in result_dict.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1).fillna(0)
    df["metric_STD"] = df.apply(lambda x: x[x["metric"] + "_STD"], axis=1).fillna(0)
    df.to_csv(f"GSG/results/csv/{name.split(' (')[0]}_proc.csv", index=False)
    df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))
    df["x_axis"] = df["task"] + df["n_runs"].apply(lambda x: " (" + str(x) + ")")


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
                "n_runs": df[df["train_pct"] == train_pct]["n_runs"].mean(),
                "x_axis": "AVG (" + str(df[df["train_pct"] == train_pct]["n_runs"].mean())[:3] + ")"
            },
            ignore_index=True,
        )

    n_complete_runs = len(df[df["train_pct"] == 100])
    if n_complete_runs == 13:
        print(name)
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

    # now the same but with plotly

    px.bar(
        df,
        x="task",
        y="metric_MEAN",
        color="% of Training Data",
        # blues colormap, 4 options - light blue, medium blue, dark blue, very dark blue
        color_discrete_sequence=["#d1e5f0", "#92c5de", "#4393c3", "#2166ac"],
        barmode="group",
        title=f"{name} results - {str(df['n_runs'].mean())[:3]} seeds",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
        error_y=df["metric_STD"],
    ).update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
        yaxis_range=[0, 1],
        yaxis_tick0=0,
        yaxis_dtick=0.05,
    ).write_image(
        f"GSG/results/plots/all/{name.split(' (')[0]}.png"
    )

    # now the same but with plotly, but only for train_pct 100
    df_100 = df[df["train_pct"] == 100]
    px.bar(
        df_100,
        x="x_axis",
        y="metric_MEAN",
        color="% of Training Data",
        # blues colormap, 4 options - light blue, medium blue, dark blue, very dark blue
        color_discrete_sequence=["#2166ac"],
        barmode="group",
        title=f"{name} results",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
        error_y=df_100["metric_STD"],
    ).update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
        yaxis_range=[0, 1],
        yaxis_tick0=0,
        yaxis_dtick=0.05,
    ).update_traces(
        width=0.5
    ).write_image(
        f"GSG/results/plots/100/{name.split(' (')[0]}.png",
        scale=3
    )
    

# now, make a plot with all entries of result_dict in one plot (only train_pct 100)
# first, make a copy of result_dict
result_dict_100 = result_dict.copy()
# then, for each entry, only keep train_pct 100
for name, df in result_dict_100.items():
    # add new column based on name
    df["Model"] = name
    if name == "Full":
        df["model_nr"] = 1
    elif name == "ST-A":
        df["model_nr"] = 2
    elif name == "C-V2":
        df["model_nr"] = 3
    elif name == "ST-A Fusion":
        df["model_nr"] = 4
    elif name == "C-V2 Fusion":
        df["model_nr"] = 5
    result_dict_100[name] = df[df["train_pct"] == 100]
# then, concatenate all entries
result_dict_100 = pd.concat(result_dict_100.values())
# then, sort by task
result_dict_100 = result_dict_100.sort_values(by=["task", "model_nr"])
# then, make plot
px.bar(
    result_dict_100,
    x="task",
    y="metric_MEAN",
    color="Model",
    # 5 options: yellow, orange, red, purple, blue
    color_discrete_sequence=["wheat", "goldenrod", "darkgoldenrod", "#92c5de", "#2166ac"],
    barmode="group",
    title=f"Comparison of all models (100% of Training Data)",
    labels={"task": "Task", "metric_MEAN": "Metric Values"},
    width=1000,
    height=600,
    template="plotly_white",
    error_y=result_dict_100["metric_STD"],
).update_layout(
    xaxis_title="Task",
    yaxis_title="Metric Values",
    xaxis_tickangle=45,
    yaxis_range=[0, 1],
    yaxis_tick0=0,
    yaxis_dtick=0.05,
).write_image(
    f"GSG/results/plots/ALL_100.png",
    # sharper
    scale=3
)

# now the same but pairwise
for setup in combinations(result_dict.keys(), 2):
    result_dict_2 = result_dict_100.copy()
    # filter for setup[0] and setup[1]
    result_dict_2 = result_dict_2[result_dict_2["Model"].isin([setup[0], setup[1]])]
    # then, make plot
    px.bar(
        result_dict_2,
        x="task",
        y="metric_MEAN",
        color="Model",
        # 5 options: yellow, orange, red, purple, blue
        color_discrete_sequence=["#92c5de", "#2166ac"],
        barmode="group",
        # include seeds
        # title=f"Comparison of {setup[0]} vs {setup[1]} (100% of Training Data)",
        title=f"{setup[0]} ({str(result_dict_drop[setup[0]]['n_runs'].mean())[:3]} seeds) vs. {setup[1]} ({str(result_dict_drop[setup[1]]['n_runs'].mean())[:3]} seeds) (100% of Training Data)",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
        error_y=result_dict_2["metric_STD"],

    ).update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
        # size
        xaxis_tickfont_size=12,
        yaxis_range=[0, 1],
        yaxis_tick0=0,
        yaxis_dtick=0.05,
    ).write_image(
        f"GSG/results/differences/plots/100/{setup[0]}_vs_{setup[1]}_100.png",
        # sharper
        scale=3
    )
print("DONE!")