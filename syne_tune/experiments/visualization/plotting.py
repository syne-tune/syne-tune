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
from typing import Dict, Any, Optional, Tuple, Union, List, Iterable, Callable
from dataclasses import dataclass
import logging
import copy

import numpy as np
import pandas as pd

from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments.visualization.aggregate_results import (
    aggregate_and_errors_over_time,
)
from syne_tune.experiments.visualization.results_utils import (
    MapMetadataToSetup,
    MapMetadataToSubplot,
    DateTimeBounds,
    create_index_for_result_files,
    load_results_dataframe_per_benchmark,
    download_result_files_from_s3,
    _insert_into_nested_dict,
)
from syne_tune.try_import import try_import_visual_message

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(try_import_visual_message())

logger = logging.getLogger(__name__)


DEFAULT_AGGREGATE_MODE = "iqm_bootstrap"


def _impute_with_defaults(original, default, names: List[str]) -> Dict[str, Any]:
    result = dict()
    for name in names:
        orig_val = getattr(original, name, None)
        result[name] = getattr(default, name, None) if orig_val is None else orig_val
    return result


def _check_and_set_defaults(
    params: Dict[str, Any], default_values: List[Tuple[str, Any]]
):
    for name, def_value in default_values:
        if params[name] is None:
            assert def_value is not None, f"{name} must be given"
            params[name] = def_value


@dataclass
class SubplotParameters:
    """
    Parameters specifying an arrangement of subplots. ``kwargs`` is mandatory.

    :param nrows: Number of rows of subplot matrix
    :param ncols: Number of columns of subplot matrix
    :param titles: If given, these are titles for each column in the
        arrangement of subplots. If ``title_each_figure == True``, these
        are titles for each subplot. If ``titles`` is not given, then
        ``PlotParameters.title`` is printed on top of the leftmost column
    :param title_each_figure: See ``titles``, defaults to ``False``
    :param kwargs: Extra arguments for ``plt.subplots``, apart from "nrows" and "ncols"
    :param legend_no: Subplot indices where legend is to be shown. Defaults
        to ``[]`` (no legends shown). This is not relative to ``subplot_indices``
    :param xlims: If this is given, must be a list with one entry per subfigure.
        In this case, the global ``xlim`` is overwritten by
        ``(0, xlims[subplot_no])``. If ``subplot_indices`` is given, ``xlims``
        must have the same length, and ``xlims[j]`` refers to subplot index
        ``subplot_indices[j]`` then
    :param subplot_indices: If this is given, we only plot subfigures with indices
        in this list, and in this order. Otherwise, we plot subfigures 0, 1, 2, ...
    """

    nrows: int = None
    ncols: int = None
    titles: List[str] = None
    title_each_figure: bool = None
    kwargs: Dict[str, Any] = None
    legend_no: List[int] = None
    xlims: List[int] = None
    subplot_indices: List[int] = None

    def merge_defaults(
        self, default_params: "SubplotParameters"
    ) -> "SubplotParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=[
                "nrows",
                "ncols",
                "titles",
                "title_each_figure",
                "kwargs",
                "legend_no",
                "xlims",
                "subplot_indices",
            ],
        )
        default_values = [
            ("nrows", None),
            ("ncols", None),
            ("title_each_figure", False),
        ]
        _check_and_set_defaults(new_params, default_values)
        return SubplotParameters(**new_params)


@dataclass
class ShowTrialParameters:
    """
    Parameters specifying the ``show_init_trials`` feature. This features adds
    one more curve to each subplot where ``setup_name`` features. This curve
    shows best metric value found for trials with ID ``<= trial_id``. The
    right-most value is extended as constant line across the remainder of the
    x-axis, for better visibility.

    :param setup_name: Setup from which the trial performance is taken
    :param trial_id: ID of trial. Defaults to 0. If this is positive, data
        from trials with IDs ``<= trial_id`` are shown
    :param new_setup_name: Name of the additional curve in legends
    """

    setup_name: str = None
    trial_id: int = None
    new_setup_name: str = None

    def merge_defaults(
        self, default_params: "ShowTrialParameters"
    ) -> "ShowTrialParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=["setup_name", "trial_id", "new_setup_name"],
        )
        default_values = [
            ("setup_name", None),
            ("new_setup_name", None),
            ("trial_id", 0),
        ]
        _check_and_set_defaults(new_params, default_values)
        return ShowTrialParameters(**new_params)


DEFAULT_XLABEL = "wall-clock time (secs)"


@dataclass
class PlotParameters:
    """
    Parameters specifying the figure.

    If ``convert_to_min == True``, then smaller is better in plots. An original
    metric value ``metric_val`` is converted as
    :code:`metric_multiplier * metric_val` if ``mode == "min"``,
    and as :code:`1 - metric_multiplier * metric_val` if ``mode == "max"``.
    If `convert_to_min == False``, we always convert as
    :code:`metric_multiplier * metric_val`, so that larger is better if
    ``mode == "max"``.

    :param metric: Name of metric, mandatory
    :param mode: See above, "min" or "max". Defaults to "min" if not given
    :param title: Title of plot. If ``subplots`` is used, see
        :class:`SubplotParameters`
    :param xlabel: Label for x axis. If ``subplots`` is used, this is
        printed below each column. Defaults to :const:`DEFAULT_XLABEL`
    :param ylabel: Label for y axis. If ``subplots`` is used, this is
        printed left of each row
    :param xlim: ``(x_min, x_max)`` for x axis. If ``subplots`` is used, see
        :class:`SubplotParameters`
    :param ylim: ``(y_min, y_max)`` for y axis.
    :param metric_multiplier: See above. Defaults to 1
    :param convert_to_min: See above. Defaults to ``True``
    :param tick_params: Params for ``ax.tick_params``
    :param aggregate_mode: How are values across seeds aggregated?

        * "mean_and_ci": Mean and 0.95 normal confidence interval
        * "median_percentiles": Mean and 25, 75 percentiles
        * "iqm_bootstrap": Interquartile mean and 0.95 confidence interval
          based on the bootstrap variance estimate

        Defaults to :const:`DEFAULT_AGGREGATE_MODE`
    :param dpi: Resolution of figure in DPI. Defaults to 200
    :param grid: Figure with grid? Defaults to ``False``
    :param subplots: If given, the figure consists of several subplots. See
         :class:`SubplotParameters`
    :param show_init_trials: See :class:`ShowTrialParameters`
    """

    metric: str = None
    mode: str = None
    title: str = None
    xlabel: str = None
    ylabel: str = None
    xlim: Tuple[float, float] = None
    ylim: Tuple[float, float] = None
    metric_multiplier: float = None
    convert_to_min: bool = None
    tick_params: Dict[str, Any] = None
    aggregate_mode: str = None
    dpi: int = None
    grid: bool = None
    subplots: SubplotParameters = None
    show_init_trials: ShowTrialParameters = None

    def merge_defaults(self, default_params: "PlotParameters") -> "PlotParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=[
                "metric",
                "mode",
                "title",
                "xlabel",
                "ylabel",
                "xlim",
                "ylim",
                "metric_multiplier",
                "convert_to_min",
                "tick_params",
                "aggregate_mode",
                "dpi",
                "grid",
            ],
        )
        default_values = [
            ("metric", None),
            ("mode", "min"),
            ("title", ""),
            ("metric_multiplier", 1),
            ("convert_to_min", True),
            ("aggregate_mode", DEFAULT_AGGREGATE_MODE),
            ("dpi", 200),
            ("grid", False),
            ("xlabel", DEFAULT_XLABEL),
        ]
        _check_and_set_defaults(new_params, default_values)
        if self.subplots is None:
            new_params["subplots"] = default_params.subplots
        elif default_params.subplots is None:
            new_params["subplots"] = self.subplots
        else:
            new_params["subplots"] = self.subplots.merge_defaults(
                default_params.subplots
            )
        if self.show_init_trials is None:
            new_params["show_init_trials"] = default_params.show_init_trials
        elif default_params.show_init_trials is None:
            new_params["show_init_trials"] = self.show_init_trials
        else:
            new_params["show_init_trials"] = self.show_init_trials.merge_defaults(
                default_params.show_init_trials
            )
        return PlotParameters(**new_params)


DataFrameColumnGenerator = Callable[[pd.DataFrame], pd.Series]


DataFrameGroups = Dict[Tuple[int, str], List[Tuple[str, pd.DataFrame]]]


def group_results_dataframe(df: pd.DataFrame) -> DataFrameGroups:
    result = dict()
    for (subplot_no, setup_name, tuner_name), tuner_df in df.groupby(
        ["subplot_no", "setup_name", "tuner_name"]
    ):
        key = (int(subplot_no), setup_name)
        value = (tuner_name, tuner_df)
        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]
    return result


def filter_final_row_per_trial(grouped_dfs: DataFrameGroups) -> DataFrameGroups:
    """
    We filter rows such that only one row per trial ID remains, namely the
    one with the largest time stamp. This makes sense for single-fidelity
    methods, where reports have still been done after every epoch.
    """
    logger.info("Filtering results down to one row per trial (final result)")
    result = dict()
    for key, tuner_dfs in grouped_dfs.items():
        new_tuner_dfs = []
        for tuner_name, tuner_df in tuner_dfs:
            df_by_trial = tuner_df.groupby("trial_id")
            max_time_in_trial = df_by_trial[ST_TUNER_TIME].transform(max)
            max_time_in_trial_mask = max_time_in_trial == tuner_df[ST_TUNER_TIME]
            new_tuner_dfs.append((tuner_name, tuner_df[max_time_in_trial_mask]))
        result[key] = new_tuner_dfs
    return result


def enrich_results(
    grouped_dfs: DataFrameGroups,
    column_name: str,
    dataframe_column_generator: Optional[DataFrameColumnGenerator],
) -> DataFrameGroups:
    if dataframe_column_generator is None:
        return grouped_dfs
    logger.info("Enriching results by additional column (dataframe_column_generator)")
    result = dict()
    for key, tuner_dfs in grouped_dfs.items():
        new_tuner_dfs = []
        for tuner_name, tuner_df in tuner_dfs:
            assert column_name not in tuner_df.columns, (
                f"New column to be appended to results dataframe: {column_name} is "
                f"already a column: {tuner_df.columns}"
            )
            new_column = dataframe_column_generator(tuner_df)
            new_tuner_dfs.append(
                (tuner_name, tuner_df.assign(**{column_name: new_column}))
            )
        result[key] = new_tuner_dfs
    return result


class ComparativeResults:
    """
    This class loads, processes, and plots results of a comparative study,
    combining several experiments for different methods, seeds, and
    benchmarks (optional). Note that an experiment corresponds to one run
    of HPO, resulting in files :const:`~syne_tune.constants.ST_METADATA_FILENAME`
    for metadata, and :const:`~syne_tune.constants.ST_RESULTS_DATAFRAME_FILENAME`
    for time-stamped results.

    There is one comparative plot per benchmark (aggregation of results
    across benchmarks are not supported here). Results are grouped by
    setup (which usually equates to method), and then summary statistics are
    shown for each setup as function of wall-clock time. The plot can also
    have several subplots, in which case results are first grouped into
    subplot number, then setup.

    If ``benchmark_key is None``, there is only a single benchmark, and all
    results are merged together.

    Both setup name and subplot number (optional) can be configured by the
    user, as function of metadata written for each experiment. The functions
    ``metadata_to_setup`` and ``metadata_to_subplot`` (optional) can also be
    used for filtering: results of experiments for which any of them returns
    ``None``, are not used.

    When grouping results w.r.t. benchmark name and setup name, we should end
    up with ``num_runs`` experiments. These are (typically) random repetitions
    with different seeds. If after grouping, a different number of experiments
    is found for some setup, a warning message is printed. In this case, we
    recommend to check the completeness of result files. Common reasons:

    * Less than ``num_runs`` experiments found. Experiments failed, or files
      were not properly synced.
    * More than ``num_runs`` experiments found. This happens if initial
      experiments for the study failed, but ended up writing results. This can
      be fixed by either removing the result files, or by using
      ``datetime_bounds`` (since initial failed experiments ran first).

    Result files have the path
    ``f"{experiment_path()}{ename}/{patt}/{ename}-*/"``, where ``path`` is from
    ``with_subdirs``, and ``ename`` from ``experiment_names``. The default is
    ``with_subdirs="*"``. If ``with_subdirs`` is ``None``, result files have
    the path ``f"{experiment_path()}{ename}-*/"``. Use this if your experiments
    have been run locally.

    If ``datetime_bounds`` is given, it contains a tuple of strings
    ``(lower_time, upper_time)``, or a dictionary mapping names from
    ``experiment_names`` to such tuples. Both strings are time-stamps in the
    format :const:`~syne_tune.constants.ST_DATETIME_FORMAT` (example:
    "2023-03-19-22-01-57"), and each can be ``None`` as well. This serves to
    filter out any result whose time-stamp does not fall within the interval
    (both sides are inclusive), where ``None`` means the interval is open on
    that side. This feature is useful to filter out results of erroneous
    attempts.

    If ``metadata_keys`` is given, it contains a list of keys into the
    metadata. In this case, metadata values for these keys are extracted and
    can be retrieved with :meth:`metadata_values`. In fact,
    ``metadata_values(benchmark_name)`` returns a nested dictionary, where
    ``result[key][setup_name]`` is a list of values. If
    ``metadata_subplot_level`` is ``True`` and ``metadata_to_subplot`` is
    given, the result structure is ``result[key][setup_name][subplot_no]``.
    This should be set if different subplots share the same setup names,
    since otherwise metadata values are only grouped by setup name.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param setups: Possible values of setup names
    :param num_runs: When grouping results w.r.t. benchmark name and setup
        name, we should end up with this many experiments. See above
    :param metadata_to_setup: See above
    :param plot_params: Parameters controlling the plot. Can be overwritten
        in :meth:`plot`. See :class:`PlotParameters`
    :param metadata_to_subplot: See above. Optional
    :param benchmark_key: Key for benchmark in metadata files. Defaults to
        "benchmark". If this is ``None``, there is only a single benchmark,
        and all results are merged together
    :param with_subdirs: See above. Defaults to "*"
    :param datetime_bounds: See above
    :param metadata_keys: See above
    :param metadata_subplot_level: See above. Defaults to ``False``
    :param download_from_s3: Should result files be downloaded from S3? This
        is supported only if ``with_subdirs``
    :param s3_bucket: Only if ``download_from_s3 == True``. If not given, the
        default bucket for the SageMaker session is used
    """

    def __init__(
        self,
        experiment_names: Tuple[str, ...],
        setups: Iterable[str],
        num_runs: int,
        metadata_to_setup: MapMetadataToSetup,
        plot_params: Optional[PlotParameters] = None,
        metadata_to_subplot: Optional[MapMetadataToSubplot] = None,
        benchmark_key: Optional[str] = "benchmark",
        with_subdirs: Optional[Union[str, List[str]]] = "*",
        datetime_bounds: Optional[DateTimeBounds] = None,
        metadata_keys: Optional[List[str]] = None,
        metadata_subplot_level: bool = False,
        download_from_s3: bool = False,
        s3_bucket: Optional[str] = None,
    ):
        if download_from_s3:
            assert (
                with_subdirs is not None
            ), "Cannot download files from S3 if with_subdirs=None"
            download_result_files_from_s3(experiment_names, s3_bucket)
        result = create_index_for_result_files(
            experiment_names=experiment_names,
            metadata_to_setup=metadata_to_setup,
            metadata_to_subplot=metadata_to_subplot,
            metadata_keys=metadata_keys,
            metadata_subplot_level=metadata_subplot_level,
            benchmark_key=benchmark_key,
            with_subdirs=with_subdirs,
            datetime_bounds=datetime_bounds,
        )
        self._reverse_index = result["index"]
        assert result["setup_names"] == set(setups), (
            f"Filtered results contain setup names {result['setup_names']}, "
            f"but should contain setup names {setups}"
        )
        self._metadata_values = (
            None if metadata_keys is None else result["metadata_values"]
        )
        self._metadata_subplot_level = metadata_subplot_level and (
            metadata_to_subplot is not None
        )
        self.setups = tuple(setups)
        self.num_runs = num_runs
        self._default_plot_params = copy.deepcopy(plot_params)

    def _check_benchmark_name(self, benchmark_name: Optional[str]) -> str:
        err_msg = f"benchmark_name must be one of {list(self._reverse_index.keys())}"
        if benchmark_name is None:
            assert len(self._reverse_index) == 1, err_msg
            benchmark_name = next(iter(self._reverse_index.keys()))
        else:
            assert benchmark_name in self._reverse_index, err_msg
        return benchmark_name

    def metadata_values(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """
        The nested dictionary returned has the structure
        ``result[key][setup_name]``, or ``result[key][setup_name][subplot_no]``
        if ``metadata_subplot_level == True``.

        :param benchmark_name: Name of benchmark
        :return: Nested dictionary with meta-data values
        """
        assert self._metadata_values is not None, (
            "Specify metadata_keys when constructing ComparativeResults if you "
            "want to extract meta-data values"
        )
        benchmark_name = self._check_benchmark_name(benchmark_name)
        return self._metadata_values[benchmark_name]

    @staticmethod
    def _figure_shape(plot_params: PlotParameters) -> Tuple[int, int]:
        subplots = plot_params.subplots
        if subplots is not None:
            nrows = subplots.nrows
            ncols = subplots.ncols
        else:
            nrows = ncols = 1
        return nrows, ncols

    def _extract_result_curves_per_experiment(
        self,
        df: pd.DataFrame,
        plot_params: PlotParameters,
        extra_results_keys: Optional[List[str]],
        one_trial_special: bool,
        setup_name: str,
        subplot_no: int,
        extra_results: Dict[str, Any],
        prev_max_rt: Optional[float],
        xlim: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts curve data ``(runtimes, values)`` which go into aggregation.

        :param df: Results grouped w.r.t. subplot, setup, and experiment.
        :param plot_params: Plot parameters
        :param extra_results_keys: If given, info is written into ``extra_results``
        :param one_trial_special: Is this special iteration for plotting results of single
            trial?
        :param setup_name:
        :param subplot_no:
        :param extra_results: Dictionary written to if ``extra_results_keys`` are given
        :param prev_max_rt: Only if ``one_trial_special == True``. Largest value of
            ``runtimes`` returned for the same experiment
        :param xlim: Range for x axis
        :return: ``(runtimes, values)``
        """
        metric = plot_params.metric
        metric_multiplier = plot_params.metric_multiplier
        show_init_trials = plot_params.show_init_trials
        if one_trial_special:
            # Filter down the dataframe
            df = df[df["trial_id"] <= show_init_trials.trial_id]
        if plot_params.mode == "max":
            ys = metric_multiplier * np.array(df[metric].cummax())
            if plot_params.convert_to_min:
                ys = 1 - ys
        else:
            ys = metric_multiplier * np.array(df[metric].cummin())
        rt = np.array(df[ST_TUNER_TIME])
        if one_trial_special:
            # Extend curve to the end, so it can be seen
            rt = np.append(rt, prev_max_rt)
            ys = np.append(ys, ys[-1])
        if xlim is not None:
            # Slice w.r.t. time. Doing this here, speeds up
            # aggregation
            ind = np.logical_and(rt >= xlim[0], rt <= xlim[1])
            rt = rt[ind]
            ys = ys[ind]
        # Collect extra results
        if extra_results_keys is not None and not one_trial_special:
            if self._metadata_subplot_level:
                key_sequence = [setup_name, subplot_no]
            else:
                key_sequence = [setup_name]
            final_pos = df[ST_TUNER_TIME].idxmax()
            final_row = dict(df.loc[final_pos])
            for key in extra_results_keys:
                _insert_into_nested_dict(
                    dictionary=extra_results,
                    key_sequence=key_sequence + [key],
                    value=final_row[key],
                )
        return rt, ys

    def _aggregate_results(
        self,
        grouped_dfs: DataFrameGroups,
        plot_params: PlotParameters,
        extra_results_keys: Optional[List[str]],
    ) -> Dict[str, Any]:
        subplots = plot_params.subplots
        subplot_xlims = None if subplots is None else subplots.xlims
        fig_shape = self._figure_shape(plot_params)
        num_subplots = fig_shape[0] * fig_shape[1]
        xlim = plot_params.xlim
        aggregate_mode = plot_params.aggregate_mode
        show_init_trials = plot_params.show_init_trials
        do_show_init_trials = show_init_trials is not None
        setup_names = self.setups
        extra_results = dict()
        if do_show_init_trials:
            # Put extra name at the end
            setup_names = setup_names + (show_init_trials.new_setup_name,)

        stats = [[None] * len(setup_names) for _ in range(num_subplots)]
        for (subplot_no, setup_name), tuner_dfs in grouped_dfs.items():
            if subplot_xlims is not None:
                xlim = (0, subplot_xlims[subplot_no])
            if do_show_init_trials and show_init_trials.setup_name == setup_name:
                num_iter = 2
            else:
                num_iter = 1
            max_rt = None
            # If this setup is named as ``show_init_trials.setup_name``, we need
            # to go over the data twice. The first iteration is as usual, the
            # second extracts the information for the single trial and extends
            # the curve.
            for it in range(num_iter):
                one_trial_special = it == 1
                if one_trial_special:
                    new_setup_name = show_init_trials.new_setup_name
                    prev_max_rt = max_rt
                else:
                    new_setup_name = setup_name
                    prev_max_rt = None
                trajectories = []
                runtimes = []
                max_rt = []
                for tuner_pos, (_, tuner_df) in enumerate(tuner_dfs):
                    rt, ys = self._extract_result_curves_per_experiment(
                        df=tuner_df,
                        plot_params=plot_params,
                        extra_results_keys=extra_results_keys,
                        one_trial_special=one_trial_special,
                        setup_name=setup_name,
                        subplot_no=subplot_no,
                        extra_results=extra_results,
                        prev_max_rt=prev_max_rt[tuner_pos]
                        if one_trial_special
                        else None,
                        xlim=xlim,
                    )
                    trajectories.append(ys)
                    runtimes.append(rt)
                    max_rt.append(rt[-1])

                setup_id = setup_names.index(new_setup_name)
                if len(trajectories) > 1:
                    stats[subplot_no][setup_id] = aggregate_and_errors_over_time(
                        errors=trajectories, runtimes=runtimes, mode=aggregate_mode
                    )
                else:
                    # If there is only a single seed, we plot a curve without
                    # error bars
                    stats[subplot_no][setup_id] = {
                        "time": runtimes[0],
                        "aggregate": trajectories[0],
                    }
                if not one_trial_special:
                    if subplots is not None:
                        msg = f"[{subplot_no}, {setup_name}]: "
                    else:
                        msg = f"[{setup_name}]: "
                    msg += f"max_rt = {np.mean(max_rt):.2f} (+- {np.std(max_rt):.2f})"
                    logger.info(msg)
                    num_repeats = len(tuner_dfs)
                    if num_repeats != self.num_runs:
                        if subplots is not None:
                            part = f"subplot = {subplot_no}, "
                        else:
                            part = ""
                        tuner_names = [name for name, _ in tuner_dfs]
                        logger.warning(
                            f"{part}setup = {setup_name} has {num_repeats} repeats "
                            f"instead of {self.num_runs}:\n{tuner_names}"
                        )

        result = {"stats": stats, "setup_names": setup_names}
        if extra_results_keys is not None:
            result["extra_results"] = extra_results
        return result

    def _transform_and_aggregrate_results(
        self,
        df: pd.DataFrame,
        plot_params: PlotParameters,
        extra_results_keys: Optional[List[str]],
        dataframe_column_generator: Optional[DataFrameColumnGenerator],
        one_result_per_trial: bool,
    ) -> Dict[str, Any]:
        # Group results according to subplot, setup, and tuner (experiment)
        grouped_dfs = group_results_dataframe(df)
        # Filter down to one result per trial
        if one_result_per_trial:
            grouped_dfs = filter_final_row_per_trial(grouped_dfs)
        # If ``dataframe_column_generator`` is given, an additional column is
        # appended to the grouped dataframes
        grouped_dfs = enrich_results(
            grouped_dfs=grouped_dfs,
            column_name=plot_params.metric,
            dataframe_column_generator=dataframe_column_generator,
        )
        # Extract curves and aggregate them
        return self._aggregate_results(grouped_dfs, plot_params, extra_results_keys)

    def _plot_figure(
        self,
        stats: List[List[Dict[str, np.ndarray]]],
        plot_params: PlotParameters,
        setup_names: List[str],
    ):
        subplots = plot_params.subplots
        if subplots is not None:
            subplot_xlims = subplots.xlims
            nrows = subplots.nrows
            ncols = subplots.ncols
            subplots_kwargs = dict(
                dict() if subplots.kwargs is None else subplots.kwargs,
                nrows=nrows,
                ncols=ncols,
            )
            subplot_titles = subplots.titles
            legend_no = [] if subplots.legend_no is None else subplots.legend_no
            if not isinstance(legend_no, list):
                legend_no = [legend_no]
            title_each_figure = subplots.title_each_figure
            subplot_indices = (
                list(range(len(stats)))
                if subplots.subplot_indices is None
                else subplots.subplot_indices
            )
        else:
            subplot_xlims = None
            nrows = ncols = 1
            subplots_kwargs = dict(nrows=nrows, ncols=ncols)
            subplot_titles = None
            legend_no = [0]
            title_each_figure = False
            subplot_indices = [0]
        if subplot_titles is None:
            subplot_titles = [plot_params.title] + ["" * (ncols - 1)]
        ylim = plot_params.ylim
        xlim = plot_params.xlim  # Can be overwritten by ``subplot_xlims``
        xlabel = plot_params.xlabel
        ylabel = plot_params.ylabel
        tick_params = plot_params.tick_params

        plt.figure(dpi=plot_params.dpi)
        figsize = (5 * ncols, 4 * nrows)
        fig, axs = plt.subplots(**subplots_kwargs, squeeze=False, figsize=figsize)
        for subplot_no, subplot_index in enumerate(subplot_indices):
            stats_subplot = stats[subplot_index]
            row = subplot_no % nrows
            col = subplot_no // nrows
            ax = axs[row, col]
            # Plot curves in the order of ``setups``. Not all setups may feature in
            # each of the subplots
            for i, (curves, setup_name) in enumerate(zip(stats_subplot, setup_names)):
                if curves is not None:
                    color = f"C{i}"
                    x = curves["time"]
                    ax.plot(x, curves["aggregate"], color=color, label=setup_name)
                    if "lower" in curves:
                        ax.plot(
                            x, curves["lower"], color=color, alpha=0.4, linestyle="--"
                        )
                        ax.plot(
                            x, curves["upper"], color=color, alpha=0.4, linestyle="--"
                        )
            if subplot_xlims is not None:
                xlim = (0, subplot_xlims[subplot_no])
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if xlabel is not None and row == nrows - 1:
                ax.set_xlabel(xlabel)
            if ylabel is not None and col == 0:
                ax.set_ylabel(ylabel)
            if tick_params is not None:
                ax.tick_params(**tick_params)
            if subplot_titles is not None:
                if title_each_figure:
                    ax.set_title(subplot_titles[subplot_no])
                elif row == 0:
                    ax.set_title(subplot_titles[col])
            if plot_params.grid:
                ax.grid(True)
            if subplot_index in legend_no:
                ax.legend()
        plt.show()
        return fig, axs

    def plot(
        self,
        benchmark_name: Optional[str] = None,
        plot_params: Optional[PlotParameters] = None,
        file_name: Optional[str] = None,
        extra_results_keys: Optional[List[str]] = None,
        dataframe_column_generator: Optional[DataFrameColumnGenerator] = None,
        one_result_per_trial: bool = False,
    ) -> Dict[str, Any]:
        """
        Create comparative plot from results of all experiments collected at
        construction, for benchmark ``benchmark_name`` (if there is a single
        benchmark only, this need not be given).

        If ``plot_params.show_init_trials`` is given, the best metric value
        curve for the data from trials ``<=  plot_params.show_init_trials.trial_id``
        in a particular setup ``plot_params.show_init_trials.setup_name`` is
        shown in all subplots the setup is contained in. This is useful to
        contrast the performance of methods against the performance for one
        particular trial, for example the initial configuration (i.e., to show
        how much this can be improved upon). The final metric value of this extra
        curve is extended until the end of the horizontal range, in order to make
        it visible. The corresponding curve is labeled with
        ``plot_params.show_init_trials.new_setup_name`` in the legend.

        If ``extra_results_keys`` is given, these are column names in the result
        dataframe. For each setup and seed, we collect the values for the
        largest time stamp. We return a nested dictionary ``extra_results``, so
        that ``extra_results[setup_name][key]`` contains values (over seeds),
        where ``key`` is in ``extra_results_keys``. If ``metadata_subplot_level``
        is ``True`` and ``metadata_to_subplot`` is given, the structure is
        ``extra_results[setup_name][subplot_no][key]``.

        If ``dataframe_column_generator`` is given, it maps a result dataframe
        for a single experiment to a new column named ``plot_params.metric``.
        This is applied before computing cumulative maximum or minimum and
        aggregation over seeds. This way, we can plot derived metrics which are
        not contained in the results as columns. Note that the transformed
        dataframe is not retained.

        :param benchmark_name: Name of benchmark for which to plot results.
            Not needed if there is only one benchmark
        :param plot_params: Parameters controlling the plot. Values provided
            here overwrite values provided at construction.
        :param file_name: If given, the figure is stored in a file of this name
        :param extra_results_keys: See above, optional
        :param dataframe_column_generator: See above, optional
        :param one_result_per_trial: If ``True``, results for each experiment
            are filtered down to one row per trial (the one with the largest
            time stamp). This is useful for results from a single-fidelity
            method, where the training script reported results after every
            epoch.
        :return: Dictionary with "fig", "axs" (for further processing). If
            ``extra_results_keys``, "extra_results" entry as stated above
        """
        benchmark_name = self._check_benchmark_name(benchmark_name)
        if plot_params is None:
            plot_params = PlotParameters()
        plot_params = plot_params.merge_defaults(self._default_plot_params)
        logger.info(f"Load results for benchmark {benchmark_name}")
        results_df = load_results_dataframe_per_benchmark(
            self._reverse_index[benchmark_name]
        )
        logger.info("Aggregate results")
        aggregate_result = self._transform_and_aggregrate_results(
            df=results_df,
            plot_params=plot_params,
            extra_results_keys=extra_results_keys,
            dataframe_column_generator=dataframe_column_generator,
            one_result_per_trial=one_result_per_trial,
        )
        fig, axs = self._plot_figure(
            stats=aggregate_result["stats"],
            plot_params=plot_params,
            setup_names=aggregate_result["setup_names"],
        )
        if file_name is not None:
            fig.savefig(file_name, dpi=plot_params.dpi)
        results = {"fig": fig, "axs": axs}
        if extra_results_keys is not None:
            results["extra_results"] = aggregate_result["extra_results"]
        return results
