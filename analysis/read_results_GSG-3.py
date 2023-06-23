from itertools import combinations
import os
import warnings
import shutil

import pandas as pd
import plotly.express as px
from extract_statistics_GSG import extract_statistics


warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

BASE_DIR = "GSG/results-c/"
OUT_FILES = [
    "FULL.csv",
    "ST-A.csv",
    "ST-A-FUSION.csv",
    "ST-A-FUSION-30.csv",
    "ST-A-FUSION-30-LR1e-5.csv",
    "ST-A-FUSION-30-LR5e-4.csv",
    "ST-A-FUSION-30-static-LR5e-4.csv",
    "ST-A-FUSION-30-static-LR5e-5.csv",
    "st-a-congosition_naive-sigmoid_init-2_lr5e-2.csv",
    "st-a-congosition_naive-sigmoid_init-2_lr1e-2.csv",
    "st-a-congosition_naive-sigmoid_init-2_lr5e-3.csv",
    "st-a-congosition_naive-sigmoid_init-2_lr1e-2-cosine.csv",
    "st-a-congosition_naive-sigmoid_init-2_5e-3-cosine.csv",
    "st-a-congosition_naive-sigmoid_init-2_5e-3-uplift.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-2-uplift.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-3-uplift.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-4.csv",
    "st-a-congosition_naive-sigmoid_init-2_5e-3.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-3.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-4.csv",
    "st-a-congosition_naive-sigmoid_init-2_1e-2.csv",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-3.csv",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-2.csv",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-4.csv",
    "st-a-congosition_naive_init-avg_lr1e-2.csv",
    "st-a-congosition_naive_init-avg_lr1e-3.csv",
    "st-a-congosition_naive-clamp_init-avg_lr1e-3.csv",
    "st-a-congosition_naive-clamp_init-avg_lr1e-2.csv",
    "st-a-congosition_naive-sigmoid5_init-avg_lr1e-3.csv",
    "st-a-congosition_naive-sigmoid5_init-avg_lr1e-2.csv",
    "st-a-congosition_naive-clamp-05_lr1e-2.csv",
    "st-a-congosition_naive-clamp-05_lr5e-3.csv",
    "st-a-congosition_naive-sigmoid-05_lr1e-2.csv",
    "st-a-congosition_naive-sigmoid-05_lr5e-3.csv",
    "st-a-congosition_naive-clamp-1-avg-init_lr1e-4-difflr.csv",
    "st-a-congosition_naive-clamp-1-avg-init_lr5e-4-difflr.csv",
    "st-a-congosition_naive-clamp-1-avg-init_lr1e-2.csv",
    "st-a-congosition_naive-clamp-1-avg-init_lr5e-3.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init_LN-after-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init_LN-before-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_sigmoid_avg-init-lr1e-2-sgd.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-3.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-4.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-4.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-3.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-3.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-3.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-3.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-4.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr5e-3.csv",
    "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-3.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-es_metric.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-difflr-lr5e-3.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-cosine.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-nwl.csv",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-tanh-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_3avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout01-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout05-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout04-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale1-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale12-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_2avg-init-dropout025-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_2avg-init-dropout03-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-warmup.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-cosine.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_001.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_01.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-3.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-difflr_beta-lr1e-2.csv",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout025-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_2avg-init-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr1e-2.csv",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr5e-3.csv",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-4.csv",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-3.csv",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-BETA_elwise-first-0-lr1e-2.csv",
]

OUT_FILES = [BASE_DIR + f for f in OUT_FILES]
DIR_NAMES = [
    "full",
    "st-a",
    "st-a-fusion-GSG-FP16",
    "st-a-fusion-GSG-FP16-30",
    "st-a-fusion-GSG-FP16-30-LR1e-5",
    "st-a-fusion-GSG-FP16-30-LR5e-4",
    "st-a-fusion-GSG-FP16-30-static",
    "st-a-fusion-GSG-FP16-30-static-LR5e-5",
    "st-a-congosition_naive-sigmoid_init-2_lr5e-2",
    "st-a-congosition_naive-sigmoid_init-2_lr1e-2",
    "st-a-congosition_naive-sigmoid_init-2_lr5e-3",
    "st-a-congosition_naive-sigmoid_init-2_lr1e-2-cosine",
    "st-a-congosition_naive-sigmoid_init-2_5e-3-cosine",
    "st-a-congosition_naive-sigmoid_init-2_5e-3-uplift",
    "st-a-congosition_naive-sigmoid_init-2_1e-2-uplift",
    "st-a-congosition_naive-sigmoid_init-2_1e-3-uplift",
    "st-a-congosition_naive-sigmoid_init-2_1e-4",
    "st-a-congosition_naive-sigmoid_init-2_5e-3",
    "st-a-congosition_naive-sigmoid_init-2_1e-3",
    "st-a-congosition_naive-sigmoid_init-2_1e-4",
    "st-a-congosition_naive-sigmoid_init-2_1e-2",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-3",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-2",
    "st-a-congosition_naive-sigmoid_init-avg_lr1e-4",
    "st-a-congosition_naive_init-avg_lr1e-2",
    "st-a-congosition_naive_init-avg_lr1e-3",
    "st-a-congosition_naive-clamp_init-avg_lr1e-3",
    "st-a-congosition_naive-clamp_init-avg_lr1e-2",
    "st-a-congosition_naive-sigmoid5_init-avg_lr1e-3",
    "st-a-congosition_naive-sigmoid5_init-avg_lr1e-2",
    "st-a-congosition_naive-clamp-05_lr1e-2",
    "st-a-congosition_naive-clamp-05_lr5e-3",
    "st-a-congosition_naive-sigmoid-05_lr1e-2",
    "st-a-congosition_naive-sigmoid-05_lr5e-3",
    "st-a-congosition_naive-clamp-1-avg-init_lr1e-4-difflr",
    "st-a-congosition_naive-clamp-1-avg-init_lr5e-4-difflr",
    "st-a-congosition_naive-clamp-1-avg-init_lr1e-2",
    "st-a-congosition_naive-clamp-1-avg-init_lr5e-3",
    "st-a-congosition_naive-param_direct_clamp_avg-init_LN-after-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_avg-init_LN-before-lr1e-2",
    "st-a-congosition_naive-param_direct_sigmoid_avg-init-lr1e-2-sgd",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-3",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-4",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-4",
    "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-3",
    "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-3",
    "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-3",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-3",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-4",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr5e-3",
    "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-2",
    "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-3",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-es_metric",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-difflr-lr5e-3",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-cosine",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-nwl",
    "st-a-congosition_naive-param_elwise_clamp_avg-init-tanh-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_3avg-init-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout01-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout05-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout04-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale1-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale12-lr1e-2",
    "st-a-congosition_naive-param_elwise_2avg-init-dropout025-lr1e-2",
    "st-a-congosition_naive-param_elwise_2avg-init-dropout03-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-warmup",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-cosine",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_001",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_01",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-3",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-difflr_beta-lr1e-2",
    "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout025-lr1e-2",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2",
    "st-a-congosition_naive-param_direct_2avg-init-lr1e-2",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr1e-2",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr5e-3",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-4",
    "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-3",
    "st-a-congosition_naive-param_direct_clamp_2avg-init-BETA_elwise-first-0-lr1e-2",
]

print(len(OUT_FILES), len(DIR_NAMES))
extract_statistics(OUT_FILES, DIR_NAMES)

EXCLUDE_LIST = [
    "ln_after",
    "ln_before-after",
    "residual",
    "CUSTOM",
    "C-V6",
    "C-V7",
    # "C-V8",
    "C-V5-ln_before",
    "FUSION-ND",
    "STATIC",
    "value_init_avg",
    "value_after",
    "value_normal",
    # "LR",
    "dk",
    "C-V0-FUSION-GSG-FP16-TT-DOUBLETT",
]

compare_dict = {
    "CONGOSITION-NAVIE": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A",
            "st-a-congosition_naive-sigmoid_init-2_lr5e-2",
            "st-a-congosition_naive-sigmoid_init-2_lr1e-2",
            "st-a-congosition_naive-sigmoid_init-2_lr5e-3",
            "st-a-congosition_naive-sigmoid_init-2_lr1e-2-cosine",
            "st-a-congosition_naive-sigmoid_init-2_5e-3-cosine",
            "st-a-congosition_naive-sigmoid_init-2_5e-3-uplift",
            "st-a-congosition_naive-sigmoid_init-2_1e-2-uplift",
            "st-a-congosition_naive-sigmoid_init-2_1e-3-uplift",
            "st-a-congosition_naive-sigmoid_init-2_1e-4",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-NAIVE-2": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A",
            "st-a-congosition_naive-sigmoid_init-2_5e-3",
            "st-a-congosition_naive-sigmoid_init-2_1e-3",
            "st-a-congosition_naive-sigmoid_init-2_1e-2",
            "st-a-congosition_naive-sigmoid_init-2_1e-4",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-3",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-2",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-4",
            "st-a-congosition_naive_init-avg_lr1e-2",
            "st-a-congosition_naive_init-avg_lr1e-3",
            "st-a-congosition_naive-clamp_init-avg_lr1e-3",
            "st-a-congosition_naive-clamp_init-avg_lr1e-2",
            "st-a-congosition_naive-sigmoid5_init-avg_lr1e-3",
            "st-a-congosition_naive-sigmoid5_init-avg_lr1e-2",
            "st-a-congosition_naive-clamp-05_lr1e-2",
            "st-a-congosition_naive-clamp-05_lr5e-3",
            "st-a-congosition_naive-sigmoid-05_lr1e-2",
            "st-a-congosition_naive-sigmoid-05_lr5e-3",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-NAIVE-3": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-3",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-2",
            "st-a-congosition_naive-sigmoid_init-avg_lr1e-4",
            "st-a-congosition_naive_init-avg_lr1e-2",
            "st-a-congosition_naive_init-avg_lr1e-3",
            "st-a-congosition_naive-clamp_init-avg_lr1e-3",
            "st-a-congosition_naive-clamp_init-avg_lr1e-2",
            # "st-a-congosition_naive-clamp-05_lr1e-2",
            # "st-a-congosition_naive-clamp-05_lr5e-3",
            # "st-a-congosition_naive-sigmoid-05_lr1e-2",
            # "st-a-congosition_naive-sigmoid-05_lr5e-3",
            "st-a-congosition_naive-clamp-1-avg-init_lr1e-4-difflr",
            "st-a-congosition_naive-clamp-1-avg-init_lr5e-4-difflr",
            "st-a-congosition_naive-clamp-1-avg-init_lr1e-2",
            "st-a-congosition_naive-clamp-1-avg-init_lr5e-3",
            # "st-a-congosition_naive-param_direct_clamp_avg-init_LN-after-lr1e-2",
            # "st-a-congosition_naive-param_direct_clamp_avg-init_LN-before-lr1e-2",
            # "st-a-congosition_naive-param_direct_sigmoid_avg-init-lr1e-2-sgd",
            "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-3",
            "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-4",
            "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr5e-4",
            "st-a-congosition_naive-param_direct_clamp_avg-init-difflr-lr1e-3",
            # "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-2",
            # "st-a-congosition_naive-param_direct_clamp_avg-init-no_res-lr1e-3",
            # "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-2",
            # "st-a-congosition_naive-param_direct_clamp_avg-init-tanh-lr1e-3",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-3",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-4",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr5e-3",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-3",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout025-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr5e-3",
            # "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-4",
            # "st-a-congosition_naive-param_direct_2avg-init-dropout03-difflr_omega-lr1e-3",
        ],
        "diff_base": "ST-A-FUSION",
    },
    
    "CONGOSITION-ELWISE-1": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A-FUSION-30",
            "ST-A",
            "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-3",
            "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-4",
            "st-a-congosition_naive-param_elwise_clamp_avg-init-lr5e-3",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-3",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-es_metric",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-difflr-lr5e-3",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-cosine",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-nwl",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-tanh-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_3avg-init-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout01-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout05-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout04-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale1-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale12-lr1e-2",
            "st-a-congosition_naive-param_elwise_2avg-init-dropout025-lr1e-2",
            "st-a-congosition_naive-param_elwise_2avg-init-dropout03-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-warmup",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-cosine",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_001",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_01",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-ELWISE-2": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A-FUSION-30",
            "ST-A",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-3",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-4",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr5e-3",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_sigmoid_avg-init-lr1e-3",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-es_metric",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-difflr-lr5e-3",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-cosine",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-lr1e-2-nwl",
            # "st-a-congosition_naive-param_elwise_clamp_avg-init-tanh-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_3avg-init-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout01-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale1-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-rescale12-lr1e-2",
            # "st-a-congosition_naive-param_elwise_2avg-init-dropout025-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-warmup",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout025-lr1e-2-cosine",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_001",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2-L2_01",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-lr1e-2",
            # "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-3",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout025-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout05-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-dropout05-difflr_beta-lr1e-2",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-ELWISE-BETA-SIMPLE": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A-FUSION-30",
            "ST-A",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_elwise-first-0-difflr_beta-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-BETA_single-first-0-difflr_beta-lr1e-2",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-SINGLE-BETA-SIMPLE": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A-FUSION-30",
            "ST-A",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-BETA_elwise-first-0-lr1e-2",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "CONGOSITION-ELWISE-PRES": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A-FUSION-30",
            "ST-A",
            "st-a-congosition_naive-param_elwise_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_elwise_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_clamp_2avg-init-dropout03-lr1e-2",
            "st-a-congosition_naive-param_direct_2avg-init-dropout03-lr1e-2",
        ],
        "diff_base": "ST-A-FUSION",
    },
    "FUSION-ABLATION": {
        "exclude": [],
        "include": [],
        "include_list_exactly": [
            "ST-A-FUSION",
            "ST-A",
            "ST-A-FUSION-30",
            "ST-A-FUSION-30-LR1e-5",
            "ST-A-FUSION-30-LR5e-4",
            "ST-A-FUSION-30-static-LR5e-4",
            "ST-A-FUSION-30-static-LR5e-5",
        ],
        "diff_base": "ST-A-FUSION",
    },
}

shutil.rmtree("GSG/results-c/differences", ignore_errors=False)
os.makedirs("GSG/results-c/differences/csv", exist_ok=True)
os.makedirs("GSG/results-c/differences/plots/all", exist_ok=True)
os.makedirs("GSG/results-c/differences/plots/100", exist_ok=True)
os.makedirs("GSG/results-c/plots/all", exist_ok=True)
os.makedirs("GSG/results-c/plots/100", exist_ok=True)
os.makedirs("GSG/results-c/csv", exist_ok=True)

# This code is used to store the results of the various models in a dictionary
result_dict = {}
for file in os.listdir("GSG/results-c"):
    if file.endswith(".csv") and not any(exclude in file for exclude in EXCLUDE_LIST):
        result_dict[file.split(".")[0]] = pd.read_csv(
            os.path.join("GSG/results-c", file)
        )


task_order = [
    "cb",
    "copa",
    "wsc",
    "rte",
    "mrpc",
    "wic",
    "stsb",
    "boolq",
    "sst2",
    "qnli",
    "qqp",
    "mnli",
    "AVG",
]


# custom sorting function based on task
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


for setup in result_dict.keys():
    # result_dict[setup] = result_dict[setup].sort_values(by=["task", "train_pct"], ascending=[False, True])
    result_dict[setup] = result_dict[setup].sort_values(
        by=["task", "train_pct"],
        key=lambda x: x.map(sort_tasks),
        ascending=[True, True],
    )


result_dict_drop = {
    k: v.drop(columns=["best_seed", "seeds"]).set_index(["task", "train_pct"])
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
    if task in ["multirc", "record"]:
        metric = "f1"
    elif task == "cola":
        metric = "matthews_correlation"
    elif task == "stsb":
        metric = "pearson"
    else:
        metric = "accuracy"
    return metric


for name, df in result_dict.items():
    df["metric"] = df["task"].apply(compute_main_metric)
    df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1).fillna(0)
    df["metric_STD"] = df.apply(lambda x: x[x["metric"] + "_STD"], axis=1).fillna(0)
    df.to_csv(f"GSG/results-c/csv/{name.split(' (')[0]}_proc.csv", index=False)
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
                "x_axis": "AVG ("
                + str(df[df["train_pct"] == train_pct]["n_runs"].mean())[:3]
                + ")",
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
        yaxis_range=[
            0.5,
            1,
        ],
        yaxis_tick0=0,
        yaxis_dtick=0.05,
    ).update_traces(
        error_y_thickness=0.7,
    ).update_xaxes(
        categoryorder="array",
        categoryarray=task_order,
    ).write_image(
        f"GSG/results-c/plots/all/{name.split(' (')[0]}.png"
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
        yaxis_range=[
            0.5,
            1,
        ],
        yaxis_tick0=0,
        yaxis_dtick=0.05,
    ).update_xaxes(
        categoryorder="array",
        categoryarray=task_order,
    ).update_traces(
        error_y_thickness=0.7,
    ).update_traces(
        width=0.5
    ).write_image(
        f"GSG/results-c/plots/100/{name.split(' (')[0]}.png", scale=3
    )


# now, make a plot with all entries of result_dict in one plot (only train_pct 100)
# first, make a copy of result_dict
result_dict_100 = result_dict.copy()
# then, for each entry, only keep train_pct 100
for name, df in result_dict_100.items():
    # add new column based on name
    df["Model"] = name
    selction_100 = df[df["train_pct"] == 100]
    df = df.append(
        {
            "task": "AVG",
            "train_pct": 100,
            "metric_MEAN": df[df["train_pct"] == 100]["metric_MEAN"].mean(),
            "metric_STD": df[df["train_pct"] == 100]["metric_STD"].mean(),
            "% of Training Data": str(100),
            "n_runs": df[df["train_pct"] == 100]["n_runs"].mean(),
            "x_axis": "AVG ("
            + str(df[df["train_pct"] == 100]["n_runs"].mean())[:3]
            + ")",
            "Model": name,
        },
        ignore_index=True,
    )
    if name == "FULL":
        df["model_nr"] = 1
    elif name == "ST-A":
        df["model_nr"] = 2
    elif name == "C-V2":
        df["model_nr"] = 3
    elif name == "ST-A-FUSION":
        df["model_nr"] = 4
    elif name == "C-V2-FUSION":
        df["model_nr"] = 5
    elif name == "C-V0":
        df["model_nr"] = 6
    elif name == "C-V5":
        df["model_nr"] = 7
    elif name == "C-V5-ln_before":
        df["model_nr"] = 8
    elif name == "C-V5-ln_after":
        df["model_nr"] = 9
    else:
        df["model_nr"] = 999
    result_dict_100[name] = df[df["train_pct"] == 100]


# then, concatenate all entries
result_dict_100 = pd.concat(result_dict_100.values())
# then, sort by task
result_dict_100 = result_dict_100.sort_values(by=["model_nr"], ascending=[True])
# then, make plot
px.bar(
    result_dict_100,
    x="task",
    y="metric_MEAN",
    color="Model",
    # 5 options: yellow, orange, red, purple, blue
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    barmode="group",
    title="Comparison of all models (100% of Training Data)",
    labels={"task": "Task", "metric_MEAN": "Metric Values"},
    width=1000,
    height=600,
    template="plotly_white",
    error_y=result_dict_100["metric_STD"],
).update_layout(
    xaxis_title="Task",
    yaxis_title="Metric Values",
    xaxis_tickangle=45,
    yaxis_range=[
        0.5,
        1,
    ],
    yaxis_tick0=0,
    yaxis_dtick=0.05,
).update_traces(
    error_y_thickness=0.25,
).update_xaxes(
    categoryorder="array",
    categoryarray=task_order,
).write_image(
    "GSG/results-c/plots/ALL_100.png",
    # sharper
    scale=3,
)

# then, make plot
px.bar(
    result_dict_100,
    x="task",
    y="metric_MEAN",
    color="Model",
    # 5 options: yellow, orange, red, purple, blue
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    barmode="group",
    title="Comparison of all models (100% of Training Data)",
    labels={"task": "Task", "metric_MEAN": "Metric Values"},
    width=1000,
    height=600,
    template="plotly_white",
    error_y=result_dict_100["metric_STD"],
).update_layout(
    xaxis_title="Task",
    yaxis_title="Metric Values",
    xaxis_tickangle=45,
    yaxis_range=[
        0.5,
        1,
    ],
    yaxis_tick0=0,
    yaxis_dtick=0.05,
).update_traces(
    error_y_thickness=0.25,
).update_xaxes(
    categoryorder="array",
    categoryarray=task_order,
).write_html(
    "GSG/results-c/plots/ALL_100.html",
    # sharper
    include_plotlyjs="cdn",
)

# now the same but without "full" and any "fusion"
# use and filter result_dict_100


def plot_100(
    df: pd.DataFrame,
    name: str,
    exclude_list: list,
    include_list: list,
    include_list_exactly: list,
    diff_base: str,
):
    if len(exclude_list) == 0:
        exclude_list = ["ZZZ"]

    if len(include_list_exactly) > 0:
        df = df[df["Model"].isin(include_list_exactly)]
    else:
        df = df[
            df["Model"].str.contains("|".join(include_list))
            & ~df["Model"].str.contains("|".join(exclude_list))
        ]
        
    
    df["Model"] = df["Model"].str.replace(
        "st-a-congosition_naive-param_", ""
    )

    unique_models = sorted(df["Model"].unique().tolist())
    last_element = unique_models[-1]  # Get the last element
    unique_models.insert(
        0, last_element
    )  # Insert the last element at the first position

    df_diff = df.copy()
    for index, row in df.iterrows():
        # get task and train_pct
        task = row["task"]
        train_pct = row["train_pct"]
        model = row["Model"]
        if model == diff_base:
            continue
        # get st-a row
        st_a_row = df[
            (df["task"] == task)
            & (df["train_pct"] == train_pct)
            & (df["Model"] == diff_base)
        ]
        # get metric_MEAN of st-a
        st_a_metric = st_a_row["metric_MEAN"].values[0]
        # subtract
        df_diff.loc[
            (df_diff["task"] == task)
            & (df_diff["train_pct"] == train_pct)
            & (df_diff["Model"] == model),
            "metric_MEAN",
        ] = (
            row["metric_MEAN"] - st_a_metric
        )
    # filter for base model
    df_diff = df_diff[df_diff["Model"] != diff_base]
    
    category_order = sorted(df["Model"].unique().tolist())
    # remove ST-A-Fusion and put it back at position 1
    category_order.remove(diff_base)
    category_order.insert(1, diff_base)
    
    fig = px.bar(
        df,
        x="task",
        y="metric_MEAN",
        color="Model",
        # first element is light grey,  next dark grey, next are Plasma_r
        color_discrete_sequence=["#d9d9d9", "#4d4d4d"] + px.colors.sequential.Plasma_r,
        barmode="group",
        title=f"Comparison of {name} models (100% of Training Data)",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
        error_y=df["metric_STD"],
        category_orders={"Model": category_order},
    )

    category_boundaries = [0.5 + i for i in range(len(df["task"].unique()))]  # Calculate the category boundaries

    for boundary in category_boundaries:
        fig.add_shape(
            type="line",
            x0=boundary,
            y0=min(df["metric_MEAN"]),
            x1=boundary,
            y1=max(df["metric_MEAN"]),
            line=dict(color="black", width=0.5),
            layer="below",
        )

    fig.update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
        yaxis_range=[
            0.5,
            1,
        ],
        yaxis_tick0=0,
        yaxis_dtick=0.025,
        font=dict(size=12),
    ).update_traces(
        error_y_thickness=0.25,
    ).update_xaxes(
        categoryorder="array",
        categoryarray=task_order,
    )

    fig.write_html(f"GSG/results-c/plots/{name}_100.html", include_plotlyjs="cdn")
    fig.write_image(f"GSG/results-c/plots/{name}_100.png", width=1000, height=600)

    # DIFFERENCE PLOT
    fig = px.bar(
        df_diff,
        x="task",
        y="metric_MEAN",
        color="Model",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        barmode="group",
        title=f"Absolute difference of models vs. {diff_base} (higher means better)",
        labels={"task": "Task", "metric_MEAN": "Metric Values"},
        width=1000,
        height=600,
        template="plotly_white",
        category_orders={"Model": sorted(df_diff["Model"].unique().tolist())},
    )

    category_boundaries = [0.5 + i for i in range(len(df_diff["task"].unique()))]  # Calculate the category boundaries

    for boundary in category_boundaries:
        fig.add_shape(
            type="line",
            x0=boundary,
            y0=min(df_diff["metric_MEAN"]),
            x1=boundary,
            y1=max(df_diff["metric_MEAN"]),
            line=dict(color="black", width=0.5),
            layer="below",
        )

    fig.update_layout(
        xaxis_title="Task",
        yaxis_title="Metric Values",
        xaxis_tickangle=45,
    ).update_traces(
        error_y_thickness=0.7,
    ).update_xaxes(
        categoryorder="array",
        categoryarray=task_order,
    )

    # Save as HTML
    fig.write_html(f"GSG/results-c/plots/{name}_100_diff.html", include_plotlyjs="cdn")

    # Save as PNG
    fig.write_image(f"GSG/results-c/plots/{name}_100_diff.png", width=1000, height=600)
    print(f"Plotted {name}")



# go through all results
for name, nested in compare_dict.items():
    # plot 100
    plot_100(
        result_dict_100,
        name,
        nested["exclude"],
        nested["include"],
        nested["include_list_exactly"],
        nested["diff_base"],
    )

# # now the same but pairwise
# for setup in combinations(result_dict.keys(), 2):
#     result_dict_2 = result_dict_100.copy()
#     # filter for setup[0] and setup[1]
#     result_dict_2 = result_dict_2[result_dict_2["Model"].isin([setup[0], setup[1]])]
#     # then, make plot
#     px.bar(
#         result_dict_2,
#         x="task",
#         y="metric_MEAN",
#         color="Model",
#         # 5 options: yellow, orange, red, purple, blue
#         color_discrete_sequence=["#92c5de", "#2166ac"],
#         barmode="group",
#         # include seeds
#         # title=f"Comparison of {setup[0]} vs {setup[1]} (100% of Training Data)",
#         title=f"{setup[0]} ({str(result_dict_drop[setup[0]]['n_runs'].mean())[:3]} seeds) vs. {setup[1]} ({str(result_dict_drop[setup[1]]['n_runs'].mean())[:3]} seeds) (100% of Training Data)",
#         labels={"task": "Task", "metric_MEAN": "Metric Values"},
#         width=1000,
#         height=600,
#         template="plotly_white",
#         error_y=result_dict_2["metric_STD"],
#     ).update_layout(
#         xaxis_title="Task",
#         yaxis_title="Metric Values",
#         xaxis_tickangle=45,
#         # size
#         xaxis_tickfont_size=12,
#         yaxis_range=[
#             0.5,
#             1,
#         ],
#         yaxis_tick0=0,
#         yaxis_dtick=0.05,
#     ).update_xaxes(
#         categoryorder="array",
#         categoryarray=task_order,
#     ).write_image(
#         f"GSG/results-c/differences/plots/100/{setup[0]}_vs_{setup[1]}_100.png",
#         # sharper
#         scale=3,
#     )


# for name, df in differences.items():
#     df["metric"] = df["task"].apply(compute_main_metric)
#     df["metric_MEAN"] = df.apply(lambda x: x[x["metric"] + "_MEAN"], axis=1)
#     df["% of Training Data"] = df["train_pct"].apply(lambda x: str(x))

#     # compute average over all tasks for different train percentages
#     avg = df.groupby("train_pct").mean()["metric_MEAN"]
#     # add 1 row for each train_pct
#     for train_pct in [10, 25, 50, 100]:
#         df = df.append(
#             {
#                 "task": "AVG",
#                 "train_pct": train_pct,
#                 "metric_MEAN": avg[train_pct],
#                 "% of Training Data": str(train_pct),
#             },
#             ignore_index=True,
#         )

#     df.to_csv(f"GSG/results-c/differences/csv/{name}.csv", index=False)

#     px.bar(
#         df,
#         x="task",
#         y="metric_MEAN",
#         color="% of Training Data",
#         # blues colormap, 4 options - light blue, medium blue, dark blue, very dark blue
#         color_discrete_sequence=["#d1e5f0", "#92c5de", "#4393c3", "#2166ac"],
#         barmode="group",
#         title=f"{name.replace('_', ' ').replace('VS', 'vs.')} (higher means first is better) - {df['n_runs_0'][0], df['n_runs_1'][0]} seeds",
#         labels={"task": "Task", "metric_MEAN": "Metric Values"},
#         width=1000,
#         height=600,
#         template="plotly_white",
#     ).update_layout(
#         xaxis_title="Task",
#         yaxis_title="Metric Values",
#         xaxis_tickangle=45,
#         yaxis_range=[-0.1, 0.1],
#         yaxis_tick0=-0.1,
#         yaxis_dtick=0.01,
#     ).update_xaxes(
#         categoryorder="array",
#         categoryarray=task_order,
#     ).write_image(
#         f"GSG/results-c/differences/plots/all/{name}.png", scale=3
#     )
print("DONE!")
