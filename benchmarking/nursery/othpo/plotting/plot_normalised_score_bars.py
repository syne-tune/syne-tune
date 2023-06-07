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
import matplotlib as mpl
import pickle
from pathlib import Path
import sys

plt.rcParams["font.size"] = 12
plt.rc("legend", fontsize=10)  # legend fontsize


mpl.rcParams["hatch.linewidth"] = 2.0
plt.rcParams["font.family"] = "Times New Roman"
sys.path.append(str(Path(__file__).parent.parent))
plt.rcParams["pdf.fonttype"] = 42

from plotting_helper import (
    colours,
    legend_order,
    task_pos_in_order,
    sort_legend_labels,
    hatches,
    get_task_values_to_plot,
)

from experiment_master_file import experiments_meta_dict, get_exp_title, get_task_values


def plot_bar(
    ax, method, label, dist, preprocessed_results, ylim_upper, task_val, it, width
):
    normalisation_score = preprocessed_results["mean_normalised_score"][method][
        task_val
    ][it - 1]
    std_normalisation_score = preprocessed_results["std_mean_normalised_score"][method][
        task_val
    ][it - 1]

    (ll,) = ax.bar(
        dist,
        min(normalisation_score, ylim_upper),
        yerr=2 * std_normalisation_score,
        error_kw={"elinewidth": 2},
        ecolor="black",
        color=colours[method],
        width=width,
        alpha=0.8,
        label=label,
        hatch=hatches[method],
        edgecolor="white",
        linewidth=0,
    )

    if normalisation_score > ylim_upper:
        # Add arrow to show that value is off plot
        ax.errorbar(
            x=[dist],
            y=[ylim_upper],
            yerr=[ylim_upper / 50],
            ecolor="black",
            capsize=3,
            capthick=0.1,
            lolims=3,
        )

    elif normalisation_score + 2 * std_normalisation_score > ylim_upper:
        ax.errorbar(
            x=[dist],
            y=[ylim_upper],
            yerr=[ylim_upper / 50],
            ecolor="dimgrey",
            capsize=3,
            capthick=0.1,
            lolims=3,
            elinewidth=0,
        )

    return ll


def prepare_ax(axes, it, exp_idx, experiments_list, experiments_meta_dict):
    ax = axes[exp_idx]
    ax.spines[["right", "top"]].set_visible(False)
    if exp_idx == len(axes) - 1:
        ax.set_xlabel("Task number")

    experiments_meta_data = experiments_meta_dict[experiments_list[exp_idx]]

    backend = experiments_meta_data["backend"]
    task_values = get_task_values(experiments_meta_data)
    task_values_to_plot = get_task_values_to_plot(backend, task_values)

    task_labels = (
        [""]
        + [str(task_pos_in_order(val, task_values)) for val in task_values_to_plot]
        + [""]
    )
    ax.set_xticklabels(task_labels)

    ax.set_ylabel(
        "%s, %s it \n Normalised score" % (get_exp_title(experiments_meta_data), it)
    )

    return ax, task_values_to_plot


def plot_bars(it, ylim_upper):

    fig, ax_bars_mult = plt.subplots(3, 1, figsize=(8, 6))

    experiments_list = ["SimOpt", "XGBoost", "YAHPO_auc_svm_1220"]
    label_list = []
    anchors = []

    extra_width = ["BoundingBox", "BoTorchTransfer"]
    width = 0.08

    for exp_idx, experiment in enumerate(experiments_list):
        experiment = experiments_list[exp_idx]

        preprocessed_results = pickle.load(
            open("plotting/preprocessed_results_%s.p" % experiment, "rb")
        )
        num_methods = len(preprocessed_results["methods"])
        sorted_methods = sorted(
            preprocessed_results["methods"], key=lambda xx: legend_order[xx]
        )

        delta_large = 1 - width * (num_methods + 0.5)

        ax_bars, task_values_to_plot = prepare_ax(
            ax_bars_mult, it, exp_idx, experiments_list, experiments_meta_dict
        )

        dist = -0.5 * width * (num_methods + 1.5)
        for task_val_idx, task_val in enumerate(task_values_to_plot):
            add_label = task_val_idx == 0 and exp_idx == 0

            for method in sorted_methods:
                dist += width
                if method in extra_width:
                    dist += 0.25 * width

                label = method if add_label else None
                ll = plot_bar(
                    ax_bars,
                    method,
                    label,
                    dist,
                    preprocessed_results,
                    ylim_upper[experiment],
                    task_val,
                    it,
                    width,
                )

                if add_label:
                    label_list.append(method)
                    anchors.append(ll)

            dist += delta_large

        low_y, high_y = ax_bars.get_ylim()
        ax_bars.set_ylim(0, min(high_y, 1.06 * ylim_upper[experiment]))

    # Add legend
    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    ordered_handles, ordered_labels = sort_legend_labels(label_list, anchors)
    ax_bars_mult[-1].legend(
        handles=ordered_handles,
        labels=ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        fancybox=False,
        shadow=False,
        ncol=4,
        frameon=False,
    )

    # Save file
    fig.align_ylabels()
    fig.tight_layout()

    plot_file = "plotting/Figures/Normalised_scores_it_%s.pdf" % it
    plt.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
    print("Generated file %s" % plot_file)


ylim_upper = {
    1: {"SimOpt": 500, "XGBoost": 400, "YAHPO_auc_svm_1220": 400},
    10: {"SimOpt": 400, "XGBoost": 100, "YAHPO_auc_svm_1220": 150},
    25: {"SimOpt": 100, "XGBoost": 30, "YAHPO_auc_svm_1220": 30},
}
for it in [1, 10, 25]:
    plot_bars(it, ylim_upper[it])
