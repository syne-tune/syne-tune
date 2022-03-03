import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional, Set

import pandas as pd

from dataclasses import dataclass

from syne_tune.constants import ST_TUNER_TIME, ST_TUNER_CREATION_TIMESTAMP
from syne_tune.tuner import Tuner
from syne_tune.util import experiment_path


@dataclass
class ExperimentResult:
    name: str
    results: pd.DataFrame
    metadata: Dict
    tuner: Tuner

    def __str__(self):
        res = f"Experiment {self.name}"
        if self.results is not None:
            res += f" contains {len(self.results)} evaluations from {len(self.results.trial_id.unique())} trials"
        if self.tuner is not None:
            metrics = ", ".join(self.tuner.scheduler.metric_names())
            res += f" when tuning {metrics} on {self.entrypoint_name()} with {self.scheduler_name()}."
        return res

    def creation_date(self):
        return datetime.fromtimestamp(self.metadata[ST_TUNER_CREATION_TIMESTAMP])

    def plot(self):
        import matplotlib.pyplot as plt
        scheduler = self.tuner.scheduler
        metric = self.metric_name()
        df = self.results
        df = df.sort_values(ST_TUNER_TIME)
        x = df.loc[:, ST_TUNER_TIME]
        y = df.loc[:, metric].cummax() if self.metric_mode() == "max" else df.loc[:, metric].cummin()
        plt.plot(x, y)
        plt.xlabel("wallclock time")
        plt.ylabel(metric)
        plt.title(self.entrypoint_name() + "-" + scheduler.__class__.__name__)

    def metric_mode(self):
        return self.tuner.scheduler.metric_mode()

    def metric_name(self) -> str:
        return self.tuner.scheduler.metric_names()[0]

    def entrypoint_name(self) -> str:
        return Path(self.tuner.backend.entrypoint_path()).stem

    def scheduler_name(self) -> str:
        return self.tuner.scheduler.__class__.__name__


def load_experiment(tuner_name: str) -> ExperimentResult:
    path = experiment_path(tuner_name)

    metadata_path = path / "metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None
    try:
        if (path / "results.csv.zip").exists():
            results = pd.read_csv(path / "results.csv.zip")
        else:
            results = pd.read_csv(path / "results.csv")
    except FileNotFoundError:
        results = None
    try:
        tuner = Tuner.load(path)
    except FileNotFoundError:
        tuner = None
    except Exception:
        tuner = None

    return ExperimentResult(
        name=tuner.name if tuner is not None else path.stem,
        results=results,
        tuner=tuner,
        metadata=metadata,
    )


def list_experiments(experiment_filter: Callable[[ExperimentResult], bool] = None) -> List[str]:
    for path in experiment_path().rglob("*/results.csv*"):
        exp = load_experiment(path.parent.name)
        if experiment_filter is None or experiment_filter(exp):
            if exp.results is not None and exp.tuner is not None and exp.metadata is not None:
                yield exp


def scheduler_name(scheduler):
    from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
    from syne_tune.optimizer.schedulers.ray_scheduler import RayTuneScheduler

    if isinstance(scheduler, FIFOScheduler):
        scheduler_name = f"SMT-{scheduler.__class__.__name__}"
        searcher = scheduler.searcher.__class__.__name__
        return "-".join([scheduler_name, searcher])
    elif isinstance(scheduler, RayTuneScheduler):
        scheduler_name = f"Ray-{scheduler.scheduler.__class__.__name__}"
        searcher = scheduler.searcher.__class__.__name__
        return "-".join([scheduler_name, searcher])
    else:
        return scheduler.__class__.__name__


def load_experiments_df(experiment_filter: Callable[[ExperimentResult], bool] = None) -> pd.DataFrame:
    """
    :param experiment_filter: only experiment where the filter is True are kept, default to None and returns everything.
    :return: a dataframe that contains all evaluations reported by tuners according to the filter given.
    The columns contains trial-id, hyperparameter evaluated, metrics observed by `report`:
     metrics collected automatically by sagemaker-tune:
     `smt_worker_time` (indicating time spent in the worker when report was seen)
     `time` (indicating wallclock time measured by the tuner)
     `decision` decision taken by the scheduler when observing the result
     `status` status of the trial that was shown to the tuner
     `config_{xx}` configuration value for the hyperparameter {xx}
     `tuner_name` named passed when instantiating the Tuner
     `entry_point_name`/`entry_point_path` name and path of the entry point that was tuned
    """
    dfs = []
    for experiment in list_experiments(experiment_filter=experiment_filter):
        assert experiment.tuner is not None
        assert experiment.results is not None
        assert experiment.metadata is not None

        df = experiment.results
        df["tuner_name"] = experiment.name
        df["scheduler"] = scheduler_name(experiment.tuner.scheduler)
        df["backend"] = experiment.tuner.backend.__class__.__name__
        df["entry_point"] = experiment.tuner.backend.entrypoint_path()
        df["entry_point_name"] = Path(experiment.tuner.backend.entrypoint_path()).stem

        metrics = experiment.tuner.scheduler.metric_names()
        # assume error is always the first
        df["metric"] = experiment.tuner.scheduler.metric_names()[0]

        scheduler_mode = experiment.tuner.scheduler.metric_mode()
        if isinstance(scheduler_mode, str):
            df["mode"] = scheduler_mode
        else:
            df["mode"] = scheduler_mode[0]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_experiments_task(entry_point, filters):
    dfs = []
    for experiment in list_experiments():
        if Path(experiment.tuner.backend.entrypoint_path()).stem == entry_point:
            stop = False

            for f in filters:
                if f not in experiment.metadata:
                    stop = True
                    break
                if filters[f] != experiment.metadata[f]:
                    stop = True
                    break
            if stop:
                continue

            df = experiment.results
            df["tuner_name"] = experiment.name
            df["backend"] = experiment.tuner.backend.__class__.__name__
            df["entry_point"] = experiment.tuner.backend.entrypoint_path()
            df["entry_point_name"] = Path(experiment.tuner.backend.entrypoint_path()).stem

            metrics = experiment.tuner.scheduler.metric_names()
            # assume error is always the first
            df["metric"] = experiment.tuner.scheduler.metric_names()[0]

            scheduler_mode = experiment.tuner.scheduler.metric_mode()
            if isinstance(scheduler_mode, str):
                df["mode"] = scheduler_mode
            else:
                df["mode"] = scheduler_mode[0]
            dfs.append(df)

    if not dfs:
        return None
    else:
        return pd.concat(dfs, ignore_index=True)


def split_per_task(df) -> Dict[str, pd.DataFrame]:
    # split by endpoint script
    dfs = {}
    for entry_point in df.entry_point_name.unique():
        df_entry_point = df.loc[df.entry_point_name == entry_point, :].dropna(axis=1, how='all')
        if "config_dataset_name" in df_entry_point:
            for dataset in df_entry_point.loc[:, "config_dataset_name"].unique():
                dataset_mask = df_entry_point.loc[:, "config_dataset_name"] == dataset
                dfs[f"{entry_point}-{dataset}"] = df_entry_point.loc[dataset_mask, :]
        else:
            dfs[entry_point] = df_entry_point
    return dfs


def load_results_by_experiment_names(
        experiment_names: Tuple[str, ...],
        metadata_to_setup: Callable[[dict], Optional[str]],
        metadata_print_values: Optional[List[str]] = None,
        metadata_to_subplot: Optional[Callable[[dict], Optional[int]]] = None,
        verbose: bool = False) -> (Optional[pd.DataFrame], Set[str]):
    """
    Loads results into a dataframe similar to `load_experiments_task`, but with
    some differences. First, tuner names are selected by their prefixes (in
    `experiment_names`).

    Second, grouping and filtering is done only w.r.t. meta, so that only
    metadata.json and results.csv.zip are needed. This works best if experiments
    have been launched with launch_hpo.py. `metadata_to_setup` maps a metadata
    dict to a setup name or None. Runs mapped to None are to be filtered out,
    while runs mapped to the same setup name are samples for that setup, giving
    rise to statistics which are plotted for that setup. Here, multiple setups
    are compared against each other in the same plot.

    `metadata_to_subplot` is optional. If given, it serves as grouping into
    subplots. Runs mapped to None are filtered out as well.

    :param experiment_names:
    :param metadata_to_setup:
    :param metadata_print_values: If given, list of metadata keys. Values for
        these keys are collected from metadata, and lists are printed
    :param metadata_to_subplot:
    :return:
    """
    dfs = []
    setup_names = set()
    if metadata_print_values is None:
        metadata_print_values = []
    for experiment_name in experiment_names:
        pattern = experiment_name + "*/metadata.json"
        print(f"pattern = {pattern}")
        metadata_values = {k: dict() for k in metadata_print_values}
        for meta_path in experiment_path().rglob(pattern):
            tuner_path = meta_path.parent
            try:
                with open(str(meta_path), "r") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                try:
                    setup_name = metadata_to_setup(metadata)
                    if metadata_to_subplot is not None:
                        subplot_no = metadata_to_subplot(metadata)
                    else:
                        subplot_no = 0
                except BaseException as err:
                    print(f"Caught exception for {tuner_path}:\n" + str(err))
                    raise
                if setup_name is not None and subplot_no is not None:
                    try:
                        if (tuner_path / "results.csv.zip").exists():
                            df = pd.read_csv(tuner_path / "results.csv.zip")
                        else:
                            df = pd.read_csv(tuner_path / "results.csv")
                    except FileNotFoundError:
                        df = None
                    if df is None:
                        if verbose:
                           print(f"{tuner_path}: Meta-data matches filter, but "
                              "results file not found. Skipping.")
                    else:
                        df['setup_name'] = setup_name
                        setup_names.add(setup_name)
                        if metadata_to_subplot is not None:
                            df['subplot_no'] = subplot_no
                        df['tuner_name'] = tuner_path.name
                        # Add all metadata fields to the dataframe
                        for k, v in metadata.items():
                            if isinstance(v, list):
                                for i, vi in enumerate(v):
                                    df[k + '_%d' % i] = vi
                            else:
                                df[k] = v
                        dfs.append(df)
                        for name in metadata_print_values:
                            if name in metadata:
                                dct = metadata_values[name]
                                value = metadata[name]
                                if setup_name in dct:
                                    dct[setup_name].append(value)
                                else:
                                    dct[setup_name] = [value]
        if metadata_print_values:
            parts = []
            for name in metadata_print_values:
                dct = metadata_values[name]
                parts.append(f"{name}:")
                for setup_name, values in dct.items():
                    parts.append(f"  {setup_name}: {values}")
            print('\n'.join(parts))

    if not dfs:
        res_df = None
    else:
        res_df = pd.concat(dfs, ignore_index=True)
    return res_df, setup_names


if __name__ == '__main__':
    for exp in list_experiments():
        if exp.results is not None:
            print(exp)

