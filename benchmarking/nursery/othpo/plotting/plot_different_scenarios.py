# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import sys

plt.rcParams["font.size"] = 13.5

sys.path.append(str(Path(__file__).parent.parent))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42

from plotting_helper import colours, labels, legend_order, task_pos_in_order
from experiment_master_file import experiments_meta_dict, get_exp_title

marker = {"WarmBO": "o", "WarmBOShuffled": "x", "PrevBO": "*", "PrevNoBO": "x"}

xlabels = {"SimOpt": "Task num", "XGBoost": "Task num", "YAHPO": "Task num"}


it = 1

experiments = [
    "YAHPO_auc_svm_1220",
    "YAHPO_auc_aknn_4538",
    "YAHPO_auc_ranger_4154",
    "YAHPO_auc_glmnet_375",
    "YAHPO_auc_svm_458",
    "YAHPO_auc_aknn_41138",
    "YAHPO_auc_ranger_40978",
    "YAHPO_auc_glmnet_40981",
]

for optimisers_unsorted in [
    ["WarmBO", "PrevBO"],
    ["WarmBO", "WarmBOShuffled"],
    ["PrevBO", "PrevNoBO"],
]:
    optimisers = sorted(optimisers_unsorted, key=lambda xx: legend_order[xx])

    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    label_list, anchor_list = [], []

    axes[0, 0].set_ylabel("Normalised score 1st it")
    axes[1, 0].set_ylabel("Normalised score 1st it")

    for idx in range(len(experiments)):
        experiment = experiments[idx]
        backend = experiments_meta_dict[experiment]["backend"]
        ax1 = idx % 4
        ax2 = 1 if idx >= 4 else 0
        axes[1, ax1].set_xlabel(xlabels[backend])
        axes[0, ax1].set_xticklabels([])
        axes[ax2, ax1].set_xlim([1.01, 21 - 0.01])
        axes[ax2, ax1].spines[["right", "top"]].set_visible(False)

        preprocessed = pickle.load(
            open("plotting/preprocessed_results_%s.p" % experiment, "rb")
        )
        axes[ax2, ax1].set_title(get_exp_title(experiments_meta_dict[experiment]))

        for task_val in preprocessed["task_values"][1:]:

            for optimiser in optimisers:

                mean_val = preprocessed["mean_normalised_score"][optimiser][task_val]
                std_val = preprocessed["std_mean_normalised_score"][optimiser][task_val]

                task_num = task_pos_in_order(task_val, preprocessed["task_values"])

                ll = axes[ax2, ax1].errorbar(
                    task_num,
                    mean_val[it - 1],
                    yerr=2 * std_val[it - 1],
                    fmt=marker[optimiser],
                    c=colours[optimiser],
                    label=labels[optimiser],
                )

                if (
                    task_val == preprocessed["task_values"][1]
                    and experiment == experiments[0]
                ):
                    label_list.append(labels[optimiser])
                    anchor_list.append(ll)

        if optimisers_unsorted != ["WarmBO", "PrevBO"]:
            for tick in axes[ax2, ax1].get_yticklabels():
                tick.set_rotation(45)

    fig.subplots_adjust(bottom=0.2, wspace=0.23)
    axes[1, 1].legend(
        handles=anchor_list,
        labels=label_list,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(1.1, -0.25),
        fancybox=False,
        shadow=False,
        ncol=4,
    )

    plot_file = "plotting/Figures/Compare_YAHPO_scenarios_%s.pdf" % optimisers
    plt.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
    print("Generated file %s" % plot_file)
