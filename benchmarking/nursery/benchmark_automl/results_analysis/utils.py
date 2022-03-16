import os

import dill
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.nursery.benchmark_automl.baselines import Methods
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata, load_experiments_df

from syne_tune.util import catchtime

rs_color = "blue"
gp_color = "orange"
tpe_color = "red"
rea_color = "brown"
hb_bb_color = "green"
hb_ts_color = "yellow"
fifo_style = 'solid'
multifidelity_style = 'dashed'
multifidelity_style2 = 'dashdot'
transfer_style = 'dotted'


@dataclass
class MethodSyle:
    color: str
    linestyle: str
    marker: str = None


show_seeds = False
method_styles = {
    Methods.RS: MethodSyle(rs_color, fifo_style),
    Methods.TPE: MethodSyle(tpe_color, fifo_style),
    Methods.GP: MethodSyle(gp_color, fifo_style),
    Methods.REA: MethodSyle(rea_color, fifo_style),
    Methods.ASHA: MethodSyle(rs_color, multifidelity_style),
    Methods.MSR: MethodSyle(rs_color, multifidelity_style2),
    Methods.BOHB: MethodSyle(tpe_color, multifidelity_style),
    Methods.MOBSTER: MethodSyle(gp_color, multifidelity_style),
    # transfer learning
    Methods.ASHA_BB: MethodSyle(hb_bb_color, multifidelity_style, "."),
    Methods.ASHA_CTS: MethodSyle(hb_ts_color, multifidelity_style, "."),
}


@dataclass
class PlotArgs:
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None


plot_range = {
    "fcnet-naval": PlotArgs(50, 1200, 0.0, 4e-3),
    "fcnet-parkinsons": PlotArgs(0, 1200, 0.0, 0.1),
    "fcnet-protein": PlotArgs(xmin=0, xmax=1200, ymin=0.225, ymax=0.35),
    "fcnet-slice": PlotArgs(50, 1200, 0.0, 0.004),
    "nas201-ImageNet16-120": PlotArgs(1000, 21000, None, 0.8),
    "nas201-cifar10": PlotArgs(2000, 21000, 0.05, 0.15),
    "nas201-cifar100": PlotArgs(3000, 21000, 0.26, 0.35),
    "lcbench-bank-marketing": PlotArgs(0, 2000, 82, 89),
    "lcbench-KDDCup09-appetency": PlotArgs(0, 2000, 96, 100),
}


def generate_df_dict(tag=None, date_min=None, date_max=None, methods_to_show=None) -> Dict[str, pd.DataFrame]:
    # todo load one df per task would be more efficient
    def metadata_filter(metadata, benchmark=None, tag=None):
        if methods_to_show is not None and not metadata['algorithm'] in methods_to_show:
            return False
        if benchmark is not None and metadata['benchmark'] != benchmark:
            return False
        if tag is not None:
            if not isinstance(tag, list):
                tag = [tag]
            if not metadata['tag'] in tag:
                return False
        if date_min is None or date_max is None:
            return True
        else:
            date_exp = datetime.fromtimestamp(metadata['st_tuner_creation_timestamp'])
            return date_min <= date_exp <= date_max

    metadatas = get_metadata()
    if tag is not None:
        if not isinstance(tag, list):
            tag = [tag]
        metadatas = {k: v for k, v in metadatas.items() if v.get("tag") in tag}
    # only select metadatas that contain the fields we are interested in
    metadatas = {
        k: v for k, v in metadatas.items()
        if all(key in v for key in ['algorithm', 'benchmark', 'tag', 'st_tuner_creation_timestamp'])
    }
    metadata_df = pd.DataFrame(metadatas.values())
    metadata_df['creation_date'] = metadata_df['st_tuner_creation_timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    metadata_df.sort_values(by='creation_date', ascending=False)
    metadata_df = metadata_df.drop_duplicates(['algorithm', 'benchmark', 'seed'])
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
        show_seeds: bool = False,
        method_styles: Optional[Dict] = None,
        ax = None,
):
    agg_results = {}
    if len(df_task) > 0:
        metric = df_task.loc[:, 'metric_names'].values[0]
        mode = df_task.loc[:, 'metric_mode'].values[0]

        if ax is None:
            fig, ax = plt.subplots()
        for algorithm, method_style in method_styles.items():
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
                    ax.plot(
                        t, y_best,
                        color=method_style.color,
                        linestyle=method_style.linestyle,
                        marker=method_style.marker,
                        alpha=0.2
                    )
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
                color=method_style.color, alpha=0.1,
            )
            ax.plot(
                t_range, mean,
                color=method_style.color,
                linestyle=method_style.linestyle,
                marker=method_style.marker,
                label=algorithm,
            )
            agg_results[algorithm] = mean

        ax.set_xlabel("wallclock time")
        ax.set_ylabel("validation error")
        ax.legend()
        ax.set_title(title)
    return ax, t_range, agg_results


def plot_results(benchmarks_to_df, method_styles: Optional[Dict] = None, prefix: str = "", title: str = None, ax=None):
    agg_results = {}

    for benchmark, df_task in benchmarks_to_df.items():
        ax, t_range, agg_result = plot_result_benchmark(
            df_task=df_task, title=benchmark, method_styles=method_styles, show_seeds=show_seeds, ax=ax,
        )
        if title is not None:
            ax.set_title(title)
        agg_results[benchmark] = agg_result
        if benchmark in plot_range:
            plotargs = plot_range[benchmark]
            ax.set_ylim([plotargs.ymin, plotargs.ymax])
            ax.set_xlim([plotargs.xmin, plotargs.xmax])

        if ax is None:
            plt.tight_layout()
            os.makedirs("figures/", exist_ok=True)
            plt.savefig(f"figures/{prefix}{benchmark}.pdf")
            plt.show()


def compute_best_value_over_time(benchmarks_to_df, methods_to_show):
    def get_results(df_task, methods_to_show):
        seed_results = {}
        if len(df_task) > 0:
            metric = df_task.loc[:, 'metric_names'].values[0]
            mode = df_task.loc[:, 'metric_mode'].values[0]

            t_max = df_task.loc[:, ST_TUNER_TIME].max()
            t_range = np.linspace(0, t_max, 10)

            for algorithm in methods_to_show:
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

                # for each seed, find the best value at each regularly spaced time-step
                y_ranges = []
                for t, y in zip(ts, ys):
                    indices = np.searchsorted(t, t_range, side="left")
                    y_range = y[np.clip(indices, 0, len(y) - 1)]
                    y_ranges.append(y_range)

                # (num_seeds, num_time_steps)
                y_ranges = np.stack(y_ranges)

                seed_results[algorithm] = y_ranges

        # seed_results shape (num_seeds, num_time_steps)
        return t_range, seed_results

    benchmark_results = []
    for benchmark, df_task in tqdm(list(benchmarks_to_df.items())):
        # (num_seeds, num_time_steps)
        _, seed_results_dict = get_results(df_task, methods_to_show)

        shapes = [x.shape for x in seed_results_dict.values()]

        # take the minimum number of seeds in case some are missing
        min_num_seeds = min(num_seed for num_seed, num_time_steps in shapes)

        # (num_methods, num_min_seeds, num_time_steps)
        seed_results = np.stack([x[:min_num_seeds] for x in seed_results_dict.values()])

        benchmark_results.append(seed_results)

    # take the minimum number of seeds in case some are missing
    min_num_seeds = min([x.shape[1] for x in benchmark_results])
    benchmark_results = np.stack([b[:, :min_num_seeds, :] for b in benchmark_results])

    # (num_benchmarks, num_methods, num_min_seeds, num_time_steps)
    return methods_to_show, np.stack(benchmark_results)


def print_rank_table(benchmarks_to_df, methods_to_show: Optional[List[str]]):
    from sklearn.preprocessing import QuantileTransformer
    from benchmarking.nursery.benchmark_automl.results_analysis.utils import compute_best_value_over_time
    import pandas as pd

    benchmarks = ['fcnet', 'nas201', 'lcbench']
    
    rows = []
    for benchmark in benchmarks:
        benchmark_to_df = {k: v for k, v in benchmarks_to_df.items() if benchmark in k}
        if len(benchmark_to_df) == 0:
            print(f"did not find evaluations for {benchmark}")
            continue
        methods_present = next(iter(benchmark_to_df.values())).algorithm.unique()
        methods_to_show = [x for x in methods_to_show if x in methods_present]

        # (num_benchmarks, num_methods, num_min_seeds, num_time_steps)
        methods_to_show, benchmark_results = compute_best_value_over_time(benchmark_to_df, methods_to_show)

        for i, task in enumerate(benchmark_to_df.keys()):
            if "lcbench" in task:
                print(task)
                # lcbench do maximization instead of minimization, we should pass the mode instead of hardcoding this
                benchmark_results *= -1

        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        benchmark_results = benchmark_results.swapaxes(0, 1)

        # (num_methods, num_benchmarks * num_min_seeds * num_time_steps)
        ranks = QuantileTransformer().fit_transform(benchmark_results.reshape(len(benchmark_results), -1))
        # ranks_std = ranks.std(axis=-1).mean(axis=0)
        row = {'benchmark': benchmark}
        row.update(dict(zip(methods_to_show, ranks.mean(axis=-1))))
        rows.append(row)
        print(row)
    df_ranks = pd.DataFrame(rows).set_index("benchmark")
    avg_row = dict(df_ranks.mean())
    avg_row["benchmark"] = "Average"
    df_ranks = pd.DataFrame(rows + [avg_row]).set_index("benchmark")
    benchmark_names = {"fcnet": "\\FCNet{}", "nas201": "\\NASBench{}", "lcbench": "\\LCBench{}"}
    df_ranks.index = df_ranks.index.map(lambda s: benchmark_names.get(s, s))
    df_ranks.columns = df_ranks.columns.map(lambda s: "\\" + s.replace("-", "") + "{}")
    print(df_ranks.to_string())
    print(df_ranks.to_latex(float_format="%.2f", na_rep="-", escape=False))

def load_and_cache(experiment_tag: Union[str, List[str]], load_cache_if_exists: bool = True, methods_to_show=None):

    result_file = Path(f"~/Downloads/cached-results-{str(experiment_tag)}.dill").expanduser()
    if load_cache_if_exists and result_file.exists():
        with catchtime(f"loading results from {result_file}"):
            with open(result_file, "rb") as f:
                benchmarks_to_df = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        benchmarks_to_df = generate_df_dict(experiment_tag, date_min=None, date_max=None, methods_to_show=methods_to_show)
        # metrics = df.metric_names
        with open(result_file, "wb") as f:
            dill.dump(benchmarks_to_df, f)

    return benchmarks_to_df