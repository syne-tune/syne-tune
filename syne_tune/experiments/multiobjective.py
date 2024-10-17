from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from syne_tune.optimizer.schedulers.multiobjective.utils import hypervolume_cumulative


def hypervolume_indicator_column_generator(
    metrics_and_modes: List[Tuple[str, str]],
    reference_point: Optional[np.ndarray] = None,
    increment: int = 1,
):
    """
    Returns generator for new dataframe column containing the best hypervolume
    indicator as function of wall-clock time, based on the metrics in
    ``metrics_and_modes`` (metric names correspond to column names in the
    dataframe). For a metric with ``mode == "max"``, we use its negative.

    This mapping is used to create the ``dataframe_column_generator`` argument
    of :meth:`~syne_tune.experiments.ComparativeResults.plot`. Since the
    current implementation is not incremental and quite slow, if you plot
    results for single-fidelity HPO methods, it is strongly recommended to
    also use ``one_result_per_trial=True``:

    .. code:: python

       results = ComparativeResults(...)
       dataframe_column_generator = hypervolume_indicator_column_generator(
           metrics_and_modes
       )
       plot_params = PlotParameters(
           metric="hypervolume_indicator",
           mode="max",
       )
       results.plot(
           benchmark_name=benchmark_name,
           plot_params=plot_params,
           dataframe_column_generator=dataframe_column_generator,
           one_result_per_trial=True,
       )

    :param metrics_and_modes: List of ``(metric, mode)``, see above
    :param reference_point: Reference point for hypervolume computation. If not
        given, a default value is used
    :param increment: If ``> 1``, the HV indicator is linearly interpolated, this
        is faster. Defaults to 1 (no interpolation)
    :return: Dataframe column generator
    """
    assert (
        len(metrics_and_modes) > 1
    ), "Cannot compute hypervolume indicator from less than 2 metrics"
    metric_names, metric_modes = zip(*metrics_and_modes)
    metric_names = list(metric_names)
    assert all(
        mode in ["min", "max"] for mode in metric_modes
    ), f"Modes must be 'min' or 'max':\n{metrics_and_modes}"
    metric_signs = np.array([1 if mode == "min" else -1 for mode in metric_modes])

    def dataframe_column_generator(df: pd.DataFrame) -> pd.Series:
        assert all(
            name in df.columns for name in metric_names
        ), f"All metric names {metric_names} must be in df.columns = {df.columns}"
        results_array = df[metric_names].values * metric_signs.reshape((1, -1))
        hv_indicator = hypervolume_cumulative(
            results_array,
            reference_point=reference_point,
            increment=increment,
        )
        return pd.Series(hv_indicator, index=df.index)

    return dataframe_column_generator
