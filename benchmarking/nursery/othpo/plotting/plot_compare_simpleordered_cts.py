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

marker = {"WarmBO": "o", "Quantiles": "*", "BoTorchTransfer": "d"}

xlabels = {"SimOpt": "Task num", "XGBoost": "Task num", "YAHPO": "Task num"}

fig, axes = plt.subplots(2, 3, figsize=(9, 5))

optimisers_unsorted = marker.keys()
otimisers = sorted(optimisers_unsorted, key=lambda xx: legend_order[xx])

its_to_plot = [1, 10]

experiments = ["SimOpt", "XGBoost", "YAHPO_auc_svm_1220"]
label_list, anchor_list = [], []

for it_idx in range(len(its_to_plot)):
    axes[it_idx, 0].set_ylabel("Normalised score %s it" % its_to_plot[it_idx])


for idx in range(len(experiments)):
    experiment = experiments[idx]
    backend = experiments_meta_dict[experiment]["backend"]
    ax = axes[0, idx]
    axes[0, idx].spines[["right", "top"]].set_visible(False)
    axes[1, idx].spines[["right", "top"]].set_visible(False)

    axes[1, idx].set_xlabel(xlabels[backend])

    preprocessed = pickle.load(
        open("plotting/preprocessed_results_%s.p" % experiment, "rb")
    )
    ax.set_title(get_exp_title(experiments_meta_dict[experiment]))

    for task_val in preprocessed["task_values"][1:]:

        for optimiser in otimisers:

            mean_val = preprocessed["mean_normalised_score"][optimiser][task_val]
            std_val = preprocessed["std_mean_normalised_score"][optimiser][task_val]

            for it_idx in range(len(its_to_plot)):
                it = its_to_plot[it_idx]

                task_num = task_pos_in_order(task_val, preprocessed["task_values"])

                ll = axes[it_idx, idx].errorbar(
                    task_num,
                    mean_val[it - 1],
                    yerr=2 * std_val[it - 1],
                    fmt=marker[optimiser],
                    c=colours[optimiser],
                    label=labels[optimiser],
                )

            if task_val == preprocessed["task_values"][2] and backend == "YAHPO":
                label_list.append(labels[optimiser])
                anchor_list.append(ll)


fig.subplots_adjust(bottom=0.2, wspace=0.23)
axes[1, 1].legend(
    handles=anchor_list,
    labels=label_list,
    loc="upper center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.25),
    fancybox=False,
    shadow=False,
    ncol=4,
)

plot_file = "plotting/Figures/Compare_simpleordered_CTS.pdf"
plt.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
print("Generated file %s" % plot_file)
