import json
import logging
from dataclasses import dataclass
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Callable, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd

from syne_tune import Tuner
from syne_tune.constants import (
    ST_METADATA_FILENAME,
    ST_RESULTS_DATAFRAME_FILENAME,
    ST_TUNER_CREATION_TIMESTAMP,
    ST_TUNER_TIME,
)
from syne_tune.util import experiment_path, metric_name_mode

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """
    Wraps results dataframe and provides retrieval services.

    :param name: Name of experiment
    :param results: Dataframe containing results of experiment
    :param metadata: Metadata stored along with results
    :param tuner: :class:`~syne_tune.Tuner` object stored along with results
    :param path: local path where the experiment is stored
    """

    name: str
    results: pd.DataFrame
    metadata: Dict[str, Any]
    tuner: Tuner
    path: Path

    def __str__(self):
        res = f"Experiment {self.name}"
        if self.results is not None:
            res += f" contains {len(self.results)} evaluations from {len(self.results.trial_id.unique())} trials"
        return res

    def creation_date(self):
        """
        :return: Timestamp when :class:`~syne_tune.Tuner` was created
        """
        return datetime.fromtimestamp(self.metadata[ST_TUNER_CREATION_TIMESTAMP])

    def plot_hypervolume(
        self,
        metrics_to_plot: Union[List[int], List[str]] = None,
        reference_point: np.ndarray = None,
        figure_path: str = None,
        **plt_kwargs,
    ):
        """Plot best hypervolume value as function of wallclock time

        :param reference_point: Reference point for hypervolume calculations.
            If None, the maximum values of each metric is used.
        :param figure_path: If specified, defines the path where the figure will be saved.
            If None, the figure is shown
        :param plt_kwargs: Arguments to :func:`matplotlib.pyplot.plot`
        """
        if metrics_to_plot is None:
            metrics_to_plot = self.metric_names()

        assert (
            len(metrics_to_plot) > 1
        ), "Only one metric defined, cannot compute hypervolume"

        metric_names, metric_modes = zip(
            *[self._metric_name_mode(metric) for metric in metrics_to_plot]
        )
        assert np.all(
            [metric_mode == "max" for metric_mode in metric_modes]
        ), f"All metrics must be maximized but the following modes were selected: {metric_modes}"

        if self.results is not None and len(self.results) > 0:
            import matplotlib.pyplot as plt
            from syne_tune.optimizer.schedulers.multiobjective.utils import (
                hypervolume_cumulative,
            )

            results_df = self.results.sort_values(ST_TUNER_TIME)

            x = self.results.loc[:, ST_TUNER_TIME]
            results_array = results_df[list(metric_names)].values
            hypervolume_indicator = hypervolume_cumulative(
                results_array, reference_point
            )
            fig, ax = plt.subplots()
            ax.plot(x, hypervolume_indicator, **plt_kwargs)
            ax.set_xlabel("wallclock time (secs)")
            ax.set_ylabel("Hypervolume indicator")
            ax.set_title(f"Hypervolume over time {self.name}")
            if figure_path is not None:
                fig.savefig(figure_path)
            else:
                fig.show()

    def plot(
        self, metric_to_plot: Union[str, int] = 0, figure_path: str = None, **plt_kwargs
    ):
        """Plot best metric value as function of wallclock time

        :param metric_to_plot: Indicates which metric to plot, can be the index or a name of the metric.
            default to 0 - first metric defined
        :param figure_path: If specified, defines the path where the figure will be saved.
            If None, the figure is shown
        :param plt_kwargs: Arguments to :func:`matplotlib.pyplot.plot`
        """
        import matplotlib.pyplot as plt

        metric_name, metric_mode = self._metric_name_mode(metric_to_plot)

        df = self.results
        if df is not None and len(df) > 0:
            df = df.sort_values(ST_TUNER_TIME)
            x = df.loc[:, ST_TUNER_TIME]
            y = (
                df.loc[:, metric_name].cummax()
                if metric_mode == "max"
                else df.loc[:, metric_name].cummin()
            )
            fig, ax = plt.subplots()
            ax.plot(x, y, **plt_kwargs)
            ax.set_xlabel("wallclock time (secs)")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Best result over time {self.name}")
            if figure_path is not None:
                fig.savefig(figure_path)
            else:
                fig.show()

    def plot_trials_over_time(
        self, metric_to_plot: Union[str, int] = 0, figure_path: str = None, figsize=None
    ):
        """Plot trials results over as function of wallclock time

        :param metric_to_plot: Indicates which metric to plot, can be the index or a name of the metric.
            default to 0 - first metric defined
        :param figure_path: If specified, defines the path where the figure will be saved.
            If None, the figure is shown
        :param figsize: width and height of figure
        """
        import matplotlib.pyplot as plt

        metric_name, metric_mode = self._metric_name_mode(metric_to_plot)
        df = self.results

        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize else (12, 4))
        for trial_id in sorted(df.trial_id.unique()):
            df_trial = df[df.trial_id == trial_id]
            df_trial.plot(
                x=ST_TUNER_TIME,
                y=metric_name,
                marker=".",
                ax=ax,
                legend=None,
                alpha=0.5,
            )
        df_stop = df[df["st_decision"] == "STOP"]
        plt.scatter(
            df_stop[ST_TUNER_TIME], df_stop[metric_name], marker="x", color="red"
        )
        plt.xlabel("Wallclock time (s)")
        plt.ylabel(metric_name)
        plt.title("Trials value over time")
        if figure_path is not None:
            fig.savefig(figure_path)
        else:
            fig.show()

    def metric_mode(self) -> Union[str, List[str]]:
        return self.metadata["metric_mode"]

    def metric_names(self) -> List[str]:
        return self.metadata["metric_names"]

    def entrypoint_name(self) -> str:
        return self.metadata["entrypoint"]

    def best_config(self, metric: Union[str, int] = 0) -> Dict[str, Any]:
        """
        Return the best config found for the specified metric
        :param metric: Indicates which metric to use, can be the index or a name of the metric.
            default to 0 - first metric defined in the Scheduler
        :return: Configuration corresponding to best metric value
        """
        metric_name, metric_mode = self._metric_name_mode(metric)

        # locate best result
        if metric_mode == "min":
            best_index = self.results.loc[:, metric_name].argmin()
        else:
            best_index = self.results.loc[:, metric_name].argmax()
        res = dict(self.results.loc[best_index])

        # Don't include internal fields
        return {k: v for k, v in res.items() if not k.startswith("st_")}

    def _metric_name_mode(self, metric: Union[str, int]) -> Tuple[str, str]:
        """
        Determine the name and its mode given ambiguous input.
        :param metric: Index or name of the selected metric
        """
        return metric_name_mode(
            metric_names=self.metric_names(),
            metric_mode=self.metric_mode(),
            metric=metric,
        )


def load_experiment(
    tuner_name: str,
    load_tuner: bool = False,
    local_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> ExperimentResult:
    """Load results from an experiment

    :param tuner_name: Name of a tuning experiment previously run
    :param load_tuner: Whether to load the tuner in addition to metadata and results
    :param local_path: Path containing the experiment to load. If not specified,
        ``~/{SYNE_TUNE_FOLDER}/`` is used.
    :param experiment_name: If given, this is used as first directory.
    :return: Result object
    """
    path = experiment_path(tuner_name, local_path)
    metadata_path = path / ST_METADATA_FILENAME
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None
    try:
        results_fname = ST_RESULTS_DATAFRAME_FILENAME
        if (path / results_fname).exists():
            results = pd.read_csv(path / results_fname)
        else:
            results = pd.read_csv(path / results_fname[:-4])
    except Exception:
        results = None
    if load_tuner:
        try:
            tuner = Tuner.load(str(path))
        except FileNotFoundError:
            tuner = None
        except Exception:
            tuner = None
    else:
        tuner = None
    return ExperimentResult(
        name=tuner.name if tuner is not None else path.stem,
        results=results,
        tuner=tuner,
        metadata=metadata,
        path=path,
    )


PathFilter = Callable[[str], bool]


ExperimentFilter = Callable[[ExperimentResult], bool]


PathOrExperimentFilter = Union[PathFilter, ExperimentFilter]


def _impute_filter(filt: Optional[PathOrExperimentFilter]) -> PathOrExperimentFilter:
    if filt is None:

        def filt(path) -> bool:
            return True

    return filt


def get_metadata(
    path_filter: Optional[PathFilter] = None, root: Path = experiment_path()
) -> Dict[str, dict]:
    """Load meta-data for a number of experiments

    :param path_filter: If passed then only experiments whose path matching
        the filter are kept. This allows rapid filtering in the presence of many
        experiments.
    :param root: Root path for experiment results. Default is
        ``experiment_path()``
    :return: Dictionary from tuner name to metadata dict
    """
    path_filter = _impute_filter(path_filter)
    res = dict()
    for metadata_path in root.glob(f"**/{ST_METADATA_FILENAME}"):
        path = metadata_path.parent
        if path_filter(str(path)):
            try:
                tuner_name = path.name
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    # we check that the metadata is valid by verifying that is a dict containing Syne Tune time-stamp
                    if (
                        isinstance(metadata, dict)
                        and ST_TUNER_CREATION_TIMESTAMP in metadata
                    ):
                        metadata["path"] = str(path.parent)
                        res[tuner_name] = metadata
            except JSONDecodeError:
                print(f"Could not read {path}")
                pass
    return res


def list_experiments(
    path_filter: Optional[PathFilter] = None,
    experiment_filter: Optional[ExperimentFilter] = None,
    root: Path = experiment_path(),
    load_tuner: bool = False,
) -> List[ExperimentResult]:
    """List experiments for which results are found

    :param path_filter: If passed then only experiments whose path matching
        the filter are kept. This allows rapid filtering in the presence of many
        experiments.
    :param experiment_filter: Filter on :class:`ExperimentResult`, optional
    :param root: Root path for experiment results. Default is result of
        :func:`experiment_path`
    :param load_tuner: Whether to load the tuner in addition to metadata and results
    :return: List of result objects
    """
    path_filter = _impute_filter(path_filter)
    experiment_filter = _impute_filter(experiment_filter)
    res = []
    for metadata_path in root.glob(f"**/{ST_METADATA_FILENAME}"):
        path = metadata_path.parent
        tuner_name = path.name
        if path_filter(str(metadata_path)):
            result = load_experiment(
                tuner_name, load_tuner, local_path=str(path.parent)
            )
            if (
                experiment_filter(result)
                and result.results is not None
                and result.metadata is not None
            ):
                res.append(result)
    return sorted(
        res,
        key=lambda result: result.metadata.get(ST_TUNER_CREATION_TIMESTAMP, 0),
        reverse=True,
    )


def load_experiments_df(
    path_filter: Optional[PathFilter] = None,
    experiment_filter: Optional[ExperimentFilter] = None,
    root: Path = experiment_path(),
    load_tuner: bool = False,
) -> pd.DataFrame:
    """
    :param path_filter: If passed then only experiments whose path matching
        the filter are kept. This allows rapid filtering in the presence of many
        experiments.
    :param experiment_filter: Filter on :class:`ExperimentResult`
    :param root: Root path for experiment results. Default is
        :func:`experiment_path`
    :param load_tuner: Whether to load the tuner in addition to metadata and results
    :return: Dataframe that contains all evaluations reported by tuners according
        to the filter given. The columns contain trial-id, hyperparameter
        evaluated, metrics reported via :class:`~syne_tune.Reporter`. These metrics
        are collected automatically:

        * ``st_worker_time`` (indicating time spent in the worker when report was
          seen)
        * ``time`` (indicating wallclock time measured by the tuner)
        * ``decision`` decision taken by the scheduler when observing the result
        * ``status`` status of the trial that was shown to the tuner
        * ``config_{xx}`` configuration value for the hyperparameter ``{xx}``
        * ``tuner_name`` named passed when instantiating the Tuner
        * ``entry_point_name``, ``entry_point_path`` name and path of the entry
          point that was tuned
    """
    dfs = []
    for experiment in list_experiments(
        path_filter=path_filter,
        experiment_filter=experiment_filter,
        root=root,
        load_tuner=load_tuner,
    ):
        assert experiment.results is not None
        assert experiment.metadata is not None

        df = experiment.results
        df["tuner_name"] = experiment.name
        for k, v in experiment.metadata.items():
            if isinstance(v, List):
                if len(v) > 1:
                    for i, x in enumerate(v):
                        df[f"{k}-{i}"] = x
                else:
                    df[k] = v[0]
            else:
                df[k] = v
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    for exp in list_experiments():
        if exp.results is not None:
            print(exp)
