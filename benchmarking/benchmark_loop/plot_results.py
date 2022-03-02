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
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import sagemaker
from matplotlib import cm
import numpy as np

from syne_tune.constants import ST_TUNER_TIME, SYNE_TUNE_FOLDER
from syne_tune.experiments import load_experiments_df
import matplotlib.pyplot as plt


def show_results(df_task, title: str, colors: Dict, show_seeds: bool = False):

    if len(df_task) > 0:
        metric = df_task.loc[:, 'metric_names'].values[0]
        mode = df_task.loc[:, 'metric_mode'].values[0]

        fig, ax = plt.subplots()

        for algorithm in sorted(df_task.algorithm.unique()):
            ts = []
            ys = []

            df_scheduler = df_task[df_task.algorithm == algorithm]
            for i, tuner_name in enumerate(df_scheduler.tuner_name.unique()):
                sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
                sub_df = sub_df.sort_values(ST_TUNER_TIME)
                t = sub_df.loc[:, ST_TUNER_TIME].values
                y_best = sub_df.loc[:, metric].cummax().values if mode == 'max' else sub_df.loc[:, metric].cummin().values
                if show_seeds:
                    ax.plot(t, y_best, color=colors[algorithm], alpha=0.2)
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
                color=colors[algorithm], alpha=0.1,
            )
            ax.plot(t_range, mean, color=colors[algorithm], label=algorithm)

        ax.set_xlabel("wallclock time")
        ax.set_ylabel(metric)
        ax.legend()
        ax.set_title(title)

    (Path(__file__).parent / "figures").mkdir(exist_ok=True)
    plt.savefig(f"figures/{title}.png")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=True,
        help="the experiment tag that was displayed when running launch_rl_benchmark.py"
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    print(f"In case you ran experiments remotely, we assume that you pulled your results by running in a terminal: \n"          
          f"aws s3 sync s3://{sagemaker.Session().default_bucket()}/{SYNE_TUNE_FOLDER}/{experiment_tag}/ ~/syne-tune/")
    experiment_filter = lambda exp: exp.metadata.get("tag") == experiment_tag
    name_filter = lambda path: experiment_tag in path
    df = load_experiments_df(name_filter, experiment_filter)
    benchmarks = df.benchmark.unique()

    for benchmark in benchmarks:
        df_task = df.loc[df.benchmark == benchmark, :]
        cmap = cm.Set3
        colors = {algorithm: cmap(i) for i, algorithm in enumerate(df.algorithm.unique())}
        show_results(df_task=df_task, title=benchmark, colors=colors)