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
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from syne_tune.optimizer.schedulers.multiobjective.utils import hypervolume_cumulative


def hypervolume_indicator_column_generator(
    metrics_and_modes: List[Tuple[str, str]],
    reference_point: Optional[np.ndarray] = None,
):
    """
    Returns generator for new dataframe column containing the best hypervolume
    indicator as function of wall-clock time, based on the metrics in
    ``metrics_and_modes`` (metric names correspond to column names in the
    dataframe). For a metric with ``mode == "min"``, we use its negative.

    :param metrics_and_modes: List of ``(metric, mode)``, see above
    :param reference_point: Reference point for hypervolume computation. If not
        given, a default value is used
    :return: Dataframe columm generator
    """
    assert (
        len(metrics_and_modes) > 1
    ), "Cannot compute hypervolume indicator from less than 2 metrics"
    metric_names, metric_modes = zip(*metrics_and_modes)
    metric_names = list(metric_names)
    assert all(
        mode in ["min", "max"] for mode in metric_modes
    ), f"Modes must be 'min' or 'max':\n{metrics_and_modes}"
    metric_signs = np.array([-1 if mode == "min" else 1 for mode in metric_modes])

    def dataframe_column_generator(df: pd.DataFrame) -> pd.Series:
        assert all(
            name in df.columns for name in metric_names
        ), f"All metric names {metric_names} must be in df.columns = {df.columns}"
        results_array = df[metric_names].values * metric_signs.reshape((1, -1))
        hv_indicator = hypervolume_cumulative(results_array, reference_point)
        return pd.Series(hv_indicator, index=df.index)

    return dataframe_column_generator
