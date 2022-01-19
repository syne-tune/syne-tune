# %%
import logging
import os
from argparse import ArgumentParser

import dill
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata, load_experiments_df


# %%

def generate_df_dict(tag: None, date_min=None, date_max=None, methods_to_show=None) -> Dict[str, pd.DataFrame]:
    # todo load one df per task would be more efficient
    def metadata_filter(metadata, benchmark=None, tag=None):
        date_exp = datetime.fromtimestamp(metadata['st_tuner_creation_timestamp'])
        if methods_to_show is not None and not metadata.get('algorithm') in methods_to_show:
            return False

        if benchmark is not None and metadata.get('benchmark') != benchmark:
            return False
        if tag is not None and metadata.get('tag') != tag:
            return False
        return date_min <= date_exp <= date_max

    metadatas = get_metadata()
    metadata_df = pd.DataFrame(metadatas.values())
    metadata_df['creation_date'] = metadata_df['st_tuner_creation_timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    creation_dates_min_max = metadata_df.groupby(['algorithm']).agg(['min', 'max'])['creation_date']
    print("creation date per method:\n" + creation_dates_min_max.to_string())

    count_per_seed = metadata_df.groupby(['algorithm', 'benchmark', 'seed']).count()['tag'].unstack()
    print("num seeds per methods: \n" + count_per_seed.to_string())

    num_seed_per_method = metadata_df.groupby(['algorithm', 'benchmark']).count()['tag'].unstack()
    print("seeds present: \n" + num_seed_per_method.to_string())

    benchmarks = list(sorted(metadata_df.benchmark.dropna().unique()))

    benchmark_to_df = {}

    for benchmark in tqdm(benchmarks):
        valid_exps = set([name for name, metadata in metadatas.items() if metadata_filter(metadata, benchmark, tag)])
        if len(valid_exps) > 0:
            def name_filter(path):
                tuner_name = Path(path).parent.stem
                return tuner_name in valid_exps
            df = load_experiments_df(name_filter)
            benchmark_to_df[benchmark] = df

    return benchmark_to_df


def plot_result_benchmark(
        df_task,
        title: str,
        colors: Dict,
        show_seeds: bool = False,
        methods_to_show: Optional[List[str]] = None,
):
    agg_results = {}
    if len(df_task) > 0:
        metric = df_task.loc[:, 'metric_names'].values[0]
        mode = df_task.loc[:, 'metric_mode'].values[0]

        fig, ax = plt.subplots()
        if methods_to_show is None:
            methods_to_show = sorted(df_task.algorithm.unique())
        for algorithm in methods_to_show:
            ts = []
            ys = []

            df_scheduler = df_task[df_task.algorithm == algorithm]
            if len(df_scheduler) == 0:
                continue
            for i, tuner_name in enumerate(df_scheduler.tuner_name.unique()):
                sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
                sub_df = sub_df.sort_values(ST_TUNER_TIME)
                t = sub_df.loc[:, ST_TUNER_TIME].values
                y_best = sub_df.loc[:, metric].cummax().values if mode == 'max' else sub_df.loc[:,
                                                                                     metric].cummin().values
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
            agg_results[algorithm] = mean

        ax.set_xlabel("wallclock time")
        ax.set_ylabel(metric)
        ax.legend()
        ax.set_title(title)
    return ax, t_range, agg_results


@dataclass
class PlotArgs:
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None


def plot_results(benchmarks_to_df, methods_to_show: Optional[List[str]] = None):
    plot_range = {
        "fcnet-naval": PlotArgs(50, None, 0.0, 4e-3),
        "fcnet-parkinsons": PlotArgs(0, None, 0.0, 0.1),
        "fcnet-protein": PlotArgs(xmin=0, ymin=None, ymax=0.35),
        "fcnet-slice": PlotArgs(50, None, 0.0, 0.004),
        "nas201-ImageNet16-120": PlotArgs(1000, None, None, 0.8),
        "nas201-cifar10": PlotArgs(2000, None, 0.05, 0.15),
        "nas201-cifar100": PlotArgs(3000, None, 0.26, 0.35),
    }
    agg_results = {}

    for benchmark, df_task in benchmarks_to_df.items():
        cmap = cm.Set3
        colors = {algorithm: cmap(i) for i, algorithm in enumerate(sorted(df_task.algorithm.unique()))}

        args = dict(df_task=df_task, title=benchmark, colors=colors, methods_to_show=methods_to_show)

        ax, t_range, agg_result = plot_result_benchmark(**args)
        agg_results[benchmark] = agg_result
        if benchmark in plot_range:
            plotargs = plot_range[benchmark]
            ax.set_ylim([plotargs.ymin, plotargs.ymax])
            ax.set_xlim([plotargs.xmin, plotargs.xmax])

        plt.tight_layout()
        plt.savefig(f"figures/{benchmark}.png")
        plt.show()


def print_rank_table(benchmarks_to_df, methods_to_show: Optional[List[str]] = None):

    def get_results(df_task):
        seed_results = {}
        if len(df_task) > 0:
            metric = df_task.loc[:, 'metric_names'].values[0]
            mode = df_task.loc[:, 'metric_mode'].values[0]

            for algorithm in sorted(df_task.algorithm.unique()):
                ts = []
                ys = []

                df_scheduler = df_task[df_task.algorithm == algorithm]
                for i, tuner_name in enumerate(df_scheduler.tuner_name.unique()):
                    sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
                    sub_df = sub_df.sort_values(ST_TUNER_TIME)
                    t = sub_df.loc[:, ST_TUNER_TIME].values
                    y_best = sub_df.loc[:, metric].cummax().values if mode == 'max' else sub_df.loc[:,
                                                                                         metric].cummin().values
                    ts.append(t)
                    ys.append(y_best)

                # compute the mean/std over time-series of different seeds at regular time-steps
                # start/stop at respectively first/last point available for all seeds
                t_min = max(tt[0] for tt in ts)
                t_max = min(tt[-1] for tt in ts)
                if t_min > t_max:
                    continue
                t_range = np.linspace(t_min, t_max, 10)

                # find the best value at each regularly spaced time-step from t_range
                y_ranges = []
                for t, y in zip(ts, ys):
                    indices = np.searchsorted(t, t_range, side="left")
                    y_range = y[indices]
                    y_ranges.append(y_range)
                y_ranges = np.stack(y_ranges)

                seed_results[algorithm] = y_ranges

        # seed_results shape (num_seeds, num_time_steps)
        return t_range, seed_results

    avg_ranks = {}
    for benchmark, df_task in tqdm(list(benchmarks_to_df.items())):

        # (num_seeds, num_time_steps)
        _, seed_results_dict = get_results(df_task)

        shapes = [x.shape for x in seed_results_dict.values()]

        # take the minimum number of seeds in case some are missing
        min_num_seeds = min(num_seed for num_seed, num_time_steps in shapes)

        # (num_methods, num_min_seeds, num_time_steps)
        seed_results = np.stack([x[:min_num_seeds] for x in seed_results_dict.values()])

        num_methods = len(seed_results)
        seed_results = seed_results.reshape(num_methods, -1)

        # (num_methods, num_min_seeds, num_time_steps)
        ranks = QuantileTransformer().fit_transform(seed_results)
        ranks = ranks.reshape(num_methods, min_num_seeds, -1)

        # (num_methods, num_time_steps)
        avg_rank = ranks.mean(axis=1)
        avg_ranks[benchmark] = avg_rank

    # %%

    methods_df = sorted(df_task.algorithm.unique())
    df_avg_ranks = pd.DataFrame(
        np.stack(list(avg_ranks.values())).mean(axis=-1),
        index=benchmarks_to_df.keys(),
        columns=methods_df,
    )
    if methods_to_show is None:
        methods_to_show = methods_df
    else:
        methods_to_show = [x for x in methods_to_show if x in methods_df]
    print(df_avg_ranks[methods_to_show].mean().to_string())


if __name__ == '__main__':
    date_min = datetime.fromisoformat("2022-01-04")
    date_max = datetime.fromisoformat("2023-01-04")

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default="cobalt-bumblebee",
        help="the experiment tag that was displayed when running the experiment"
    )
    args, _ = parser.parse_known_args()
    tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache = True
    methods_to_show = [
        'RS', 'GP', 'HB', 'MOBSTER', 'RS-MSR',
        'RS-BB', 'HB-BB']

    result_file = Path("~/Downloads/cached-results.dill").expanduser()
    if load_cache and result_file.exists():
        print(f"loading results from {result_file}")
        with open(result_file, "rb") as f:
            benchmarks_to_df = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        benchmarks_to_df = generate_df_dict(tag, date_min, date_max, methods_to_show)
        # metrics = df.metric_names
        with open(result_file, "wb") as f:
            dill.dump(benchmarks_to_df, f)

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}

    plot_results(benchmarks_to_df, methods_to_show)

    print_rank_table(benchmarks_to_df, methods_to_show)

