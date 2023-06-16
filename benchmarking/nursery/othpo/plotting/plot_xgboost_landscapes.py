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
import os
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_json_res

from plotting_helper import task_pos_in_order

plt.rcParams["font.size"] = 14

cbar_fontsize = 12

exp_folder = "xgboost_experiment_results/random-mnist/"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42

agg_experiments = load_json_res(exp_folder + "aggregated_experiments.json")


def plot_hyp_per(
    ax,
    x_par_raw,
    y_par_raw,
    ii_ch,
    res,
    data_sizes,
    n_mark=10,
    plot_x_label=True,
    plot_y_label=True,
    plot_title=True,
    plot_colour_bar=True,
    vmin=None,
    vmax=None,
    marker_size=100,
    fontsize=14,
):

    plt.rcParams["font.size"] = fontsize

    transl_dict = {
        "learning_rate": "learning_rates",
    }

    x_par = transl_dict[x_par_raw] if x_par_raw in transl_dict else x_par_raw
    y_par = transl_dict[y_par_raw] if y_par_raw in transl_dict else y_par_raw

    N_hyp = len(res["parameters_mat"][x_par])

    vals = {}
    for pt_idx in range(N_hyp):
        tup = (
            res["parameters_mat"][x_par][pt_idx],
            res["parameters_mat"][y_par][pt_idx],
        )
        vv = np.array(res["test_error_mat"])[ii_ch, pt_idx, 0]
        if tup in vals:
            vals[tup].append(vv)
        else:
            vals[tup] = [vv]

    if vmin is None:
        vmin = np.min(np.array(res["test_error_mat"])[ii_ch, :, 0])
    if vmax is None:
        vmax = np.max(np.array(res["test_error_mat"])[ii_ch, :, 0])

    for tup in vals:
        cb = ax.scatter(
            tup[0],
            tup[1],
            alpha=0.5,
            c=np.mean(vals[tup]),
            norm=plc.LogNorm(vmin=vmin, vmax=vmax),
            s=15,
        )

    if plot_colour_bar:
        plt.colorbar(cb, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if plot_x_label:
        ax.set_xlabel(x_par, fontsize=fontsize)
    else:
        ax.set_xticklabels([])
    if plot_y_label:
        ax.set_ylabel(y_par, fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    if plot_title:
        task_num = task_pos_in_order(data_sizes[ii_ch], data_sizes)
        ax.set_title("Task %s (%s)" % (task_num, data_sizes[ii_ch]), fontsize=fontsize)

    idx = list(range(N_hyp))
    sorted_idx = [
        x for _, x in sorted(zip(np.array(res["test_error_mat"])[ii_ch, :, 0], idx))
    ]
    ax.scatter(
        np.array(res["parameters_mat"][x_par])[sorted_idx[:n_mark]],
        np.array(res["parameters_mat"][y_par])[sorted_idx[:n_mark]],
        marker="X",
        c="k",
        s=marker_size,
        edgecolors="white",
        linewidths=1.75,
    )
    return cb


###############################################
# Plot five landscapes for paper
###############################################

vals_to_plot = [56, 335, 2012, 12064, 56000]
par_x = "max_depth"
par_y = "n_estimators"

vmin = np.min(agg_experiments["test_error_mat"])
vmax = np.max(agg_experiments["test_error_mat"])

N_plot = len(vals_to_plot)
fig, ax = plt.subplots(1, N_plot, figsize=(11, 2))
for pp in range(N_plot):
    ii_ch = np.where(agg_experiments["data_sizes"] == vals_to_plot[pp])[0][0]
    cb = plot_hyp_per(
        ax[pp],
        par_x,
        par_y,
        ii_ch,
        agg_experiments,
        agg_experiments["data_sizes"],
        plot_y_label=(pp == 0),
        plot_colour_bar=False,
        vmin=vmin,
        vmax=vmax,
        fontsize=14,
    )
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
cbar = fig.colorbar(cb, cax=cbar_ax)
cbar.set_label("# misclassified points", fontsize=cbar_fontsize)
os.makedirs("plotting/Figures/", exist_ok=True)
plot_file = "plotting/Figures/XGBoost_landscape_%s_%s.pdf" % (par_x, par_y)
plt.savefig(plot_file, dpi=400, bbox_inches="tight")
print("Generated file %s" % plot_file)
###############################################

print("Generating additional XGBoost hyperparameter landscape plots.")
os.makedirs("plotting/Figures/XGBoost-landscapes", exist_ok=True)


###############################################
# Plot hyperparameter landscapes overview
fig, ax = plt.subplots(6, 2, figsize=(10, 25))

plt.rcParams["font.size"] = 28
par_combos = [
    ("learning_rate", "max_depth"),
    ("learning_rate", "n_estimators"),
    ("learning_rate", "min_child_weight"),
    ("max_depth", "n_estimators"),
    ("max_depth", "min_child_weight"),
    ("n_estimators", "min_child_weight"),
]

label = "sagemaker"

for pp in range(len(par_combos)):
    par_x, par_y = par_combos[pp]
    plot_hyp_per(
        ax[pp, 0],
        par_x,
        par_y,
        0,
        agg_experiments,
        agg_experiments["data_sizes"],
        plot_title=(pp == 0),
        marker_size=250,
        fontsize=28,
    )
    plot_hyp_per(
        ax[pp, 1],
        par_x,
        par_y,
        27,
        agg_experiments,
        agg_experiments["data_sizes"],
        plot_y_label=False,
        plot_title=(pp == 0),
        marker_size=250,
        fontsize=28,
    )
    ax[pp, 0].tick_params(axis="x", labelsize=25)
    ax[pp, 0].tick_params(axis="y", labelsize=25)
    ax[pp, 1].tick_params(axis="x", labelsize=25)
    ax[pp, 1].tick_params(axis="y", labelsize=25)

plt.tight_layout()
plot_file = "plotting/Figures/XGBoost-landscapes/Two_by_two_parameters_%s.pdf" % label
plt.savefig(plot_file, bbox_inches="tight", pad_inches=0)
print("Generated file %s" % plot_file)

###############################################
