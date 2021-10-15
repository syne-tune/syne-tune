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
import logging
from pathlib import Path
from typing import Dict

from matplotlib import cm
import numpy as np

from sagemaker_tune.backend.sagemaker_backend.sagemaker_utils import download_sagemaker_results
from sagemaker_tune.constants import SMT_TUNER_TIME
from sagemaker_tune.experiments import load_experiments_df, split_per_task
import matplotlib.pyplot as plt


def show_results(df_task, title: str, colors: Dict, show_seeds: bool = False):

    if len(df_task) > 0:
        metric = df_task.loc[:, 'metric'].values[0]
        mode = df_task.loc[:, 'mode'].values[0]

        fig, ax = plt.subplots()

        for scheduler in sorted(df_task.scheduler.unique()):
            ts = []
            ys = []

            df_scheduler = df_task[df_task.scheduler == scheduler]
            for i, tuner_name in enumerate(df_scheduler.tuner_name.unique()):
                sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
                sub_df = sub_df.sort_values(SMT_TUNER_TIME)
                t = sub_df.loc[:, SMT_TUNER_TIME].values
                y_best = sub_df.loc[:, metric].cummax().values if mode == 'max' else sub_df.loc[:, metric].cummin().values
                if show_seeds:
                    ax.plot(t, y_best, color=colors[scheduler], alpha=0.2)
                ts.append(t)
                ys.append(y_best)

            # compute the mean/std over time-series of different seeds at regular time-steps
            # start/stop at respectively first/last point available for all seeds
            t_min = max(tt[0] for tt in ts)
            t_max = min(tt[-1] for tt in ts)
            if t_min > t_max:
                continue
            t_range = np.linspace(t_min, t_max)

            # find the best value at each regularly spaced time-step from t_range
            y_ranges = []
            for t, y in zip(ts, ys):
                indices = np.searchsorted(t, t_range, side="left")
                y_range = y[indices]
                y_ranges.append(y_range)
            y_ranges = np.stack(y_ranges)

            mean = y_ranges.mean(axis=0)
            std = y_ranges.std(axis=0)
            ax.fill_between(
                t_range, mean - std, mean + std,
                color=colors[scheduler], alpha=0.1,
            )
            ax.plot(t_range, mean, color=colors[scheduler], label=scheduler)

        ax.set_xlabel("wallclock time")
        ax.set_ylabel(metric)
        ax.legend()
        ax.set_title(title)

    (Path(__file__).parent / "figures").mkdir(exist_ok=True)
    plt.savefig(f"figures/{title}.png")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    download_sagemaker_results()
    df = load_experiments_df()
    df_per_task = split_per_task(df)
    print("evaluation recorded per endpoint script: ")
    print(df.entry_point_name.value_counts().to_string())

    cmap = cm.Set3
    colors = {scheduler: cmap(i) for i, scheduler in enumerate(df.scheduler.unique())}
    for task, df_task in df_per_task.items():
        show_results(df_task=df_task, title=task, colors=colors)