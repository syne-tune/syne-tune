import logging
from pathlib import Path

from matplotlib import cm

from benchmarking.nursery.benchmark_automl.baselines import Methods
from benchmarking.nursery.benchmark_automl.results_analysis.utils import MethodSyle, load_and_cache, plot_results

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from syne_tune.constants import ST_TUNER_TIME
from experiment_toolkit.aggregate_results import aggregate_and_errors_over_time
from experiment_toolkit.experiments_results import \
    load_results_by_experiment_names


BACKEND_SIMULATED = {'simulated', 'BlackboxRepositoryBackend'}

BACKEND_LOCAL = {'local', 'LocalBackend'}

BACKEND_SAGEMAKER = {'sagemaker', 'SagemakerBackend', 'SageMakerBackend'}


DEFAULT_AGGREGATE_MODE = 'median_percentiles'


COSTAWARE_FIFO_FINE_SETUPS = (
    'RS', 'BO', 'CABO-fine-0.5', 'CABO-fine-1')


def plot_results_for_experiment(experiment: dict, ax = None):
    num_runs = experiment['num_runs']
    xlabel = experiment.get('xlabel')
    ylabel = experiment.get('ylabel')
    tick_params = experiment.get('tick_params')
    aggregate_mode = experiment.get('aggregate_mode', DEFAULT_AGGREGATE_MODE)
    subplots = experiment.get('subplots')
    if subplots is not None:
        metadata_to_subplot = subplots['metadata_to_subplot']
        subplots_kwargs = subplots['subplots_kwargs']
        subplot_titles = subplots.get('titles')
        legend_no = subplots.get('legend_no')
    else:
        metadata_to_subplot = None
        subplots_kwargs = dict(nrows=1, ncols=1)
        subplot_titles = [experiment.get('title')]
        legend_no = 0
    nrows = subplots_kwargs['nrows']
    ncols = subplots_kwargs['ncols']
    num_subplots = nrows * ncols

    df, setup_names = load_results_by_experiment_names(
        experiment_names=experiment['experiment_names'],
        metadata_to_setup=experiment['metadata_to_setup'],
        metadata_print_values=experiment.get('metadata_print_values'),
        metadata_to_subplot=metadata_to_subplot)
    assert setup_names == set(experiment['setups']), \
        f"Filtered results contain setup names {setup_names}, but should " +\
        f"contain setup names {experiment['setups']}"

    stats = [[None] * len(setup_names) for _ in range(num_subplots)]
    if subplots is not None:
        iter_df = df.groupby('subplot_no')
    else:
        iter_df = [(0, df)]
    for subplot_no, subplot_df in iter_df:
        for setup_name, setup_df in subplot_df.groupby('setup_name'):
            traj = []
            runtime = []
            trial_nums = []
            metric = experiment['metric']
            if isinstance(metric, dict):
                metric = metric[setup_name]
            mode = experiment['mode']
            if isinstance(mode, dict):
                mode = mode[setup_name]
            tuner_names = []
            for tuner_name, sub_df in setup_df.groupby('tuner_name'):
                tuner_names.append(tuner_name)
                if mode == 'max':
                    ys = 1 - np.array(sub_df[metric].cummax())
                else:
                    ys = np.array(sub_df[metric].cummin())
                try:
                    rt = np.array(sub_df[ST_TUNER_TIME])
                except Exception:
                    # Data may be old?
                    rt = np.array(sub_df['smt_tuner_time'])
                traj.append(ys)
                runtime.append(rt)
                trial_nums.append(len(sub_df.trial_id.unique()))
            setup_id = experiment['setups'].index(setup_name)
            stats[subplot_no][setup_id] = aggregate_and_errors_over_time(
                errors=traj, runtimes=runtime, mode=aggregate_mode)
            num_repeats = len(tuner_names)
            if num_repeats != num_runs:
                if subplots is not None:
                    part = f"subplot = {subplot_no}, "
                else:
                    part = ""
                print(f"{part}setup = {setup_name} has {num_repeats} repeats "
                      f"instead of {num_runs}:\n{tuner_names}")

    # Do the plotting. First, iterate over subplots (if any)
    xlim = experiment.get('xlim')
    ylim = experiment.get('ylim')
    if ax is None:
        fig, ax = plt.subplots()
    figsize = (5 * ncols, 4 * nrows)
    for subplot_no, stats_subplot in enumerate(stats):
        # Plot curves in the order of experiment['setups']
        for i, (curves, setup_name) in enumerate(zip(
                stats_subplot, experiment['setups'])):
            color = f"C{i}"
            x = curves['time']
            ax.plot(x, curves['aggregate'], color=color, label=setup_name)
            ax.plot(x, curves['lower'], color=color, alpha=0.4, linestyle='--')
            ax.plot(x, curves['upper'], color=color, alpha=0.4, linestyle='--')
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if tick_params is not None:
            ax.tick_params(**tick_params)
        ax.set_title(subplot_titles[0])
        if subplot_no == legend_no:
            ax.legend()



COMPARE_ASHA_SIM_LOCAL_SETUPS = (
    'ASHA-stop-simul', 'ASHA-stop-local', 'ASHA-prom-simul', 'ASHA-prom-local')

def compare_asha_sim_local_setupmap(metadata: dict) -> Optional[str]:
    setup_name = None
    scheduler = metadata['scheduler']
    searcher = metadata['searcher']
    backend = metadata['backend']
    if searcher == 'random':
        part1 = 'ASHA-stop' if scheduler == 'hyperband_stopping' else 'ASHA-prom'
        part2 = 'simul' if backend in BACKEND_SIMULATED else 'local'
        setup_name = part1 + '-' + part2
    return setup_name


def plot_worker_speed(ax):
    experiment_tag = "nworkers-rebel-ibex"
    logging.getLogger().setLevel(logging.INFO)

    cmap = cm.get_cmap("viridis")
    method_styles = {
        f'{Methods.ASHA} (1 workers)': MethodSyle(cmap(0), "-"),
        f'{Methods.ASHA} (2 workers)': MethodSyle(cmap(0.25), "-"),
        f'{Methods.ASHA} (4 workers)': MethodSyle(cmap(0.5), "-"),
        f'{Methods.ASHA} (8 workers)': MethodSyle(cmap(1.0), "-"),
    }

    load_cache_if_exists = True

    methods_to_show = list(method_styles.keys())
    benchmarks_to_df = load_and_cache(
        load_cache_if_exists=load_cache_if_exists, experiment_tag=experiment_tag,
        methods_to_show=methods_to_show
    )

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}

    plot_results(
        benchmarks_to_df, method_styles, prefix="number-workers-", title="Impact of parallelism on wallclock time",
        ax=ax,
    )

if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(10, 4))

    plot_worker_speed(ax=axes[0])
    experiment = {
        'title': 'Simulation versus local backend',
        'experiment_names': ('jmlr-4',),
        'metadata_to_setup': compare_asha_sim_local_setupmap,
        'setups': COMPARE_ASHA_SIM_LOCAL_SETUPS,
        'num_runs': 10,
        'ylim': (0.26, 0.35),
        'xlim': (3000, 21000),
        'metric': 'metric_valid_error',
        'mode': 'min',
        'metadata_print_values': ['real_experiment_time'],
        'fname': 'comp_sim_local_asha_nb201_cifar100.pdf',
        'xlabel': 'wall-clock time',
        # 'ylabel': 'validation error',
        'aggregate_mode': 'mean_and_ci',
        'tick_params': {'labelsize': 8},
    }
    plot_results_for_experiment(experiment, ax=axes[1])
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / "figures/worker-speed-simulation.pdf"))
    plt.show()
