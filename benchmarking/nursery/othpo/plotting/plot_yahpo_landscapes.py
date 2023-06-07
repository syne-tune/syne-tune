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
import pandas as pd
import numpy as np
from typing import List
import sys
from pathlib import Path

plt.rcParams["font.family"] = "Times New Roman"
sys.path.append(str(Path(__file__).parent.parent))

plt.rcParams["font.size"] = 11
plt.rcParams["pdf.fonttype"] = 42

from plotting_helper import task_pos_in_order
from blackbox_helper import get_configs
from collect_yahpo_evaluations_for_plotting import hp_names as hp_names_raw


def plot_hp_perf(
    scenario: str,
    instance: str,
    metric: str,
    train_fracs: List[float],
    x_par,
    y_par,
    log_x=True,
    log_y=True,
    marker_colour="black",
    task_values=None,
    save_fig=True,
    alpha_back=1,
    alpha_crosses=1,
    fig_axes=None,
    only_crosses=False,
    edgewidth=1.75,
    cross_size=100,
):

    hp_names = [
        hp
        for hp in hp_names_raw[scenario]
        if hp not in ["trainsize", "task_id", "repl"]
    ]
    if fig_axes is None:
        fig, axes = plt.subplots(1, len(train_fracs), figsize=(8, 2 * 8 / 11))
    else:
        fig, axes = fig_axes

    csv_path = "yahpo_data/%s/%s.csv" % (scenario, instance)
    if not os.path.exists(csv_path):
        raise ValueError

    df = pd.read_csv(csv_path)

    for j, frac in enumerate(train_fracs):
        tdf_i = df[df["train_frac"] == frac]

        x = []
        for _, row in tdf_i.iterrows():
            hp = eval(row["hp_key"])
            hp.update({metric: row[metric]})
            x.append(hp)
        pdf = pd.DataFrame(x, columns=hp_names + [metric])
        perf_df = pdf.groupby([x_par, y_par])[metric].mean().reset_index()
        if not only_crosses:
            sc = axes[j].scatter(
                perf_df[x_par],
                perf_df[y_par],
                c=perf_df[metric],
                s=5,
                vmin=np.min(df[metric]),
                vmax=np.max(df[metric]),
                alpha=alpha_back,
                zorder=0,
            )

        inds = perf_df[metric].reset_index().nlargest(10, columns=metric).index
        for k, ind in enumerate(inds):
            axes[j].scatter(
                perf_df[x_par].loc[ind],
                perf_df[y_par].loc[ind],
                marker="X",
                color=marker_colour,
                s=cross_size,
                edgecolors="white",
                linewidths=edgewidth,
                alpha=alpha_crosses,
                zorder=20,
            )

        axes[j].set_xlabel(x_par)
        if log_x:
            axes[j].set_xscale("log")
        if log_y:
            axes[j].set_yscale("log")

        task_num = task_pos_in_order(int(frac * 20), task_values)
        axes[j].set_title("Task %s (%s %%)" % (task_num, int(frac * 100)))
    axes[0].set_ylabel(y_par)

    for ii in range(len(train_fracs)):
        if scenario == "rbv2_svm":
            axes[ii].set_xticks([0.01, 100])
        elif scenario == "rbv2_aknn":
            axes[ii].set_xticks([20, 40])
        elif scenario == "rbv2_ranger":
            axes[ii].set_xticks([500, 1500])
        elif scenario == "rbv2_glmnet":
            axes[ii].set_xticks([0.001, 0.1])

    if not only_crosses:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(metric)

    for ii in range(1, len(train_fracs)):
        axes[ii].set_yticklabels([])

    if save_fig:
        plot_file = f"plotting/Figures/yahpo_landscapes/yahpo_landscape_{scenario}_{instance}_{x_par}_{y_par}_2d.pdf"
        plt.savefig(plot_file, dpi=400, bbox_inches="tight", pad_inches=0)
        print("Generated file %s" % plot_file)
    return fig, axes


pars_to_plot = {
    "rbv2_svm": ("cost", "tolerance"),
    "rbv2_aknn": ("k", "M"),
    "rbv2_ranger": ("num.trees", "min.node.size"),
    "rbv2_glmnet": ("alpha", "s"),
}
train_fracs = [0.05, 0.25, 0.5, 0.75, 1.0]


if __name__ == "__main__":
    os.makedirs("plotting/Figures/yahpo_landscapes", exist_ok=True)

    instances = {
        "rbv2_svm": [1220, 458],
        "rbv2_aknn": [4538, 41138],
        "rbv2_ranger": [4154, 40978],
        "rbv2_glmnet": [375, 40981],
    }

    for scenario in ["rbv2_svm", "rbv2_aknn", "rbv2_ranger", "rbv2_glmnet"]:
        for instance in instances[scenario]:

            log_axis = {
                "rbv2_svm": (True, True),
                "rbv2_aknn": (False, False),
                "rbv2_ranger": (False, False),
                "rbv2_glmnet": (True, True),
            }[scenario]

            x_par, y_par = pars_to_plot[scenario]
            task_values = get_configs(
                "YAHPO", yahpo_dataset=instance, yahpo_scenario=scenario
            )[0]

            _ = plot_hp_perf(
                scenario,
                instance,
                metric="auc",
                train_fracs=train_fracs,
                x_par=x_par,
                y_par=y_par,
                log_x=log_axis[0],
                log_y=log_axis[1],
                marker_colour="black",
                task_values=task_values,
            )
