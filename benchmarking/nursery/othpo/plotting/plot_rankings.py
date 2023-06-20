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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

plt.rcParams["font.size"] = 13.5
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


from plotting_helper import colours, sort_legend_labels

from experiment_master_file import experiments_meta_dict, get_exp_title


def append_dicts(main_dict, sub_dict):
    for key in sub_dict:
        main_dict[key].append(sub_dict[key])


def get_rankings(preprocessed, task_id, ii):
    # Deals with ties by giving both the lowest score
    tmp = {
        method: preprocessed["mean_normalised_score"][method][task_id][ii]
        for method in preprocessed["methods"]
    }
    return {
        method: np.sum(np.array(list(tmp.values())) <= tmp[method]) for method in tmp
    }


methods_to_use = ["WarmBO", "Quantiles", "BoTorchTransfer", "PrevBO"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

label_list, anchor_list = [], []
ax_idx = 0
for file, experiment in [
    ("plotting/preprocessed_results_simopt.p", "SimOpt"),
    ("plotting/preprocessed_results_XGBoost.p", "XGBoost"),
    ("plotting/preprocessed_results_YAHPO_auc_svm_1220.p", "YAHPO_auc_svm_1220"),
]:
    ax = axes[ax_idx]
    meta_dict = experiments_meta_dict[experiment]
    ax.set_title(get_exp_title(meta_dict))
    preprocessed = pickle.load(open(file, "rb"))

    mean_ranking_dicts = {method: [] for method in preprocessed["methods"]}
    std_mean_ranking_dicts = {method: [] for method in preprocessed["methods"]}
    for ii in range(25):
        ranking_dicts = {method: [] for method in preprocessed["methods"]}
        for task_id in preprocessed["task_values"]:
            append_dicts(ranking_dicts, get_rankings(preprocessed, task_id, ii))
        for method in ranking_dicts:
            mean_ranking_dicts[method].append(np.mean(ranking_dicts[method]))
            std_mean_ranking_dicts[method].append(
                np.std(ranking_dicts[method], ddof=1)
                / np.sqrt(len(preprocessed["task_values"]))
            )

    for method in methods_to_use:
        if ax_idx == 0:
            label = method
            label_list.append(label)
        else:
            label = None
        aa = ax.plot(
            range(1, 26),
            mean_ranking_dicts[method],
            "d-",
            color=colours[method],
            linewidth=5,
            markersize=10,
            label=label,
        )
        if ax_idx == 0:
            anchor_list.append(aa[0])
        ax.plot(
            range(1, 26),
            np.array(mean_ranking_dicts[method])
            + 2 * np.array(std_mean_ranking_dicts[method]),
            ":",
            color=colours[method],
            alpha=0.3,
        )
        ax.plot(
            range(1, 26),
            np.array(mean_ranking_dicts[method])
            - 2 * np.array(std_mean_ranking_dicts[method]),
            ":",
            color=colours[method],
            alpha=0.3,
        )
        ax.fill_between(
            range(1, 26),
            np.array(mean_ranking_dicts[method])
            - 2 * np.array(std_mean_ranking_dicts[method]),
            np.array(mean_ranking_dicts[method])
            + 2 * np.array(std_mean_ranking_dicts[method]),
            color=colours[method],
            alpha=0.3,
        )
    ax.set_xlabel("Iteration")
    ax.spines[["right", "top"]].set_visible(False)
    if ax_idx == 0:
        ax.set_ylabel("Ranking")
    ax_idx += 1

# Add legend
fig.subplots_adjust(bottom=0.3, wspace=0.33)

ordered_handles, ordered_labels = sort_legend_labels(label_list, anchor_list)
axes[1].legend(
    handles=ordered_handles,
    labels=ordered_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.4),
    fancybox=False,
    shadow=False,
    ncol=4,
    frameon=False,
)

plot_file = "plotting/Figures/Rankings_over_iterations.pdf"
plt.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
print("Generated file %s" % plot_file)
