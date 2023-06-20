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
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import sys

plt.rcParams["font.size"] = 13.5
plt.rc("legend", fontsize=10)  # legend fontsize

sys.path.append(str(Path(__file__).parent.parent))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42

from plotting_helper import (
    colours,
    linestyles,
    sort_legend_labels,
    get_task_values_to_plot,
    task_pos_in_order,
)

from experiment_master_file import experiments_meta_dict, get_task_values


def plot_mode(ax, preprocessed_results, method, task_val, colour):
    mean_method = preprocessed_results[hyper_metric][method][task_val]
    std_method = preprocessed_results["std_" + hyper_metric][method][task_val]
    return ax.plot(
        range(1, preprocessed_results["points_per_task"] + 1),
        mean_method,
        c=colour,
        label=method,
        linestyle=linestyles[method],
    )


hyper_metric = "mean_normalised_score"

for experiment in ["SimOpt", "XGBoost", "YAHPO_auc_svm_1220"]:
    experiment_meta_data = experiments_meta_dict[experiment]

    preprocessed_results = pickle.load(
        open("plotting/preprocessed_results_%s.p" % experiment, "rb")
    )
    metric = preprocessed_results["metric"]
    backend = experiment_meta_data["backend"]

    task_values = get_task_values(experiment_meta_data)
    task_vals_to_plot = get_task_values_to_plot(backend, task_values)

    fig, axes = plt.subplots(1, len(task_vals_to_plot), figsize=(10, 2))
    anchors = []
    label_list = []
    for aa in range(len(task_vals_to_plot)):
        ax = axes[aa]
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")

        for method in preprocessed_results["methods"]:
            ll = plot_mode(
                ax,
                preprocessed_results,
                method,
                task_vals_to_plot[aa],
                colour=colours[method],
            )
            if aa == 0:
                anchors += ll
                label_list.append(method)
        ax.set_xlabel("Iteration")

        ax.set_title("Task %s" % task_pos_in_order(task_vals_to_plot[aa], task_values))
    if hyper_metric == "mean_normalised_score":
        axes[0].set_ylabel("Normalised score")
    else:
        axes[0].set_ylabel("Best %s" % metric)

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ordered_handles, ordered_labels = sort_legend_labels(label_list, anchors)
    axes[2].legend(
        handles=ordered_handles,
        labels=ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        fancybox=False,
        shadow=False,
        ncol=4,
        frameon=False,
    )

    os.makedirs("plotting/Figures", exist_ok=True)
    plot_file = "plotting/Figures/Iteration_curve_%s.pdf" % experiment
    plt.savefig(plot_file, bbox_inches="tight", pad_inches=0)
    print("Generated file %s" % plot_file)
