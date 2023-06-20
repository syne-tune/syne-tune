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
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from blackbox_helper import get_configs
from plotting_helper import colours, sort_legend_labels
from plot_yahpo_landscapes import plot_hp_perf, pars_to_plot
from experiment_master_file import experiments_meta_dict

plt.rcParams["font.size"] = 13.5
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


def add_samples_kde(
    ax, task_id, optimiser, res_to_plot, num_its, label, anchors, hyp_x, hyp_y, alpha
):

    res = res_to_plot[optimiser]

    xx_list, yy_list = [], []
    for seed in range(50):
        for it in range(num_its):
            xx = res[(backend, optimiser, seed)][task_id][hyp_x].loc[it]
            yy = res[(backend, optimiser, seed)][task_id][hyp_y].loc[it]
            xx_list.append(xx)
            yy_list.append(yy)
    aa = ax.scatter(
        xx_list,
        yy_list,
        c=colours[optimiser],
        label=label,
        s=50,
        zorder=10,
        alpha=alpha,
    )

    df = pd.DataFrame({hyp_x: xx_list, hyp_y: yy_list})
    sns.kdeplot(
        df,
        x=hyp_x,
        y=hyp_y,
        color=colours[optimiser],
        ax=ax,
        alpha=0.5,
        fill=True,
        zorder=0,
    )

    if label is not None:
        anchors.append(aa)

    return anchors


experiment = "YAHPO_auc_svm_1220"
experiments_meta_data = experiments_meta_dict[experiment]

scenario = experiments_meta_data["yahpo_scenario"]
dataset = experiments_meta_data["yahpo_dataset"]
backend = experiments_meta_data["backend"]

train_fracs = [0.25, 0.25, 1.0, 1.0]


opt_to_plot = [
    "Quantiles",
    "BoTorchTransfer",
    "WarmBO",
    "PrevBO",
]

res_to_plot = {key: {} for key in opt_to_plot}

hyp_x, hyp_y = pars_to_plot[scenario]

for ff in experiments_meta_data["files"]:
    res = pickle.load(open("optimisation_results/%s" % ff, "rb"))
    for key in res.keys():
        assert key[0] == backend
        optimiser = key[1]
        if optimiser in opt_to_plot:
            res_to_plot[optimiser][key] = res[key]

task_values, get_backend = get_configs(
    backend, yahpo_dataset=dataset, yahpo_scenario=scenario
)
config_space = get_backend(1)[1]

N_plot = len(opt_to_plot)
fig, axes = plt.subplots(N_plot, len(train_fracs), figsize=(12, 7))
for idx in range(N_plot):
    fig_sub, ax_sub = plot_hp_perf(
        scenario="%s" % scenario,
        instance=dataset,
        metric=experiments_meta_data["metric"],
        train_fracs=train_fracs,
        x_par=hyp_x,
        y_par=hyp_y,
        log_x=True,
        log_y=True,
        marker_colour="black",
        task_values=task_values,
        save_fig=False,
        alpha_back=0.5,
        alpha_crosses=1,
        fig_axes=(fig, axes[idx, :]),
        only_crosses=True,
        edgewidth=0,
        cross_size=50,
    )

it = 0

anchors, label_list = [], []
for ax_idx in range(N_plot):
    optimiser = opt_to_plot[ax_idx]
    label = optimiser
    label_list.append(label)
    for task_idx, task_id in enumerate([int(20 * tf) for tf in train_fracs]):
        num_its = 1 if task_idx % 2 == 0 else 10
        alpha = 0.25 if num_its == 1 else 0.15
        anchors = add_samples_kde(
            axes[ax_idx, task_idx],
            task_id,
            optimiser,
            res_to_plot,
            num_its,
            label,
            anchors,
            hyp_x,
            hyp_y,
            alpha,
        )
        label = None


for ax_idx in range(N_plot):
    for task_idx in range(len(train_fracs)):
        ax = axes[ax_idx, task_idx]
        if ax_idx < N_plot - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if ax_idx > 0:
            ax.set_title("")
        if task_idx == 0:
            ax.set_yticks([0.001, 0.1])
        if task_idx > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        if ax_idx == 0:
            if task_idx in [0, 2]:
                axes[0, task_idx].set_title(
                    axes[0, task_idx].get_title() + "\n first evaluation"
                )
            elif task_idx in [1, 3]:
                axes[0, task_idx].set_title(
                    axes[0, task_idx].get_title() + "\n first ten evaluations"
                )
        ax.set_xlim([config_space[hyp_x].lower / 3, config_space[hyp_x].upper * 3])
        ax.set_ylim([config_space[hyp_y].lower / 3, config_space[hyp_y].upper * 3])

fig.subplots_adjust(bottom=0.2, wspace=0.33)
ordered_handles, ordered_labels = sort_legend_labels(label_list, anchors)
axes[ax_idx, 2].legend(
    handles=ordered_handles,
    labels=ordered_labels,
    loc="upper center",
    bbox_to_anchor=(0.0, -0.4),
    fancybox=False,
    shadow=False,
    ncol=4,
    frameon=False,
)

fig.subplots_adjust(wspace=0.08, hspace=0.1)

plot_file = "plotting/Figures/sampling_locations.pdf"
fig.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
print("Generated file %s" % plot_file)
