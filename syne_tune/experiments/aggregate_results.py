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
from typing import Dict, List
import numpy as np
import pandas as pd
from numpy.random import default_rng
from collections import OrderedDict


def fill_trajectory(
    performance_list: List[np.ndarray], time_list: List[np.ndarray], replace_nan=np.NaN
) -> (np.ndarray, np.ndarray):
    frame_dict = OrderedDict()
    for c, (p, t) in enumerate(zip(performance_list, time_list)):
        assert len(p) == len(t), f"({c}) Array length mismatch: {len(p)} != {len(t)}"
        ds = pd.Series(data=p, index=t)
        ds = ds[~ds.index.duplicated(keep="first")]
        frame_dict[str(c)] = ds
    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()
    performance = merged.values
    time_ = merged.index.values
    performance[np.isnan(performance)] = replace_nan
    assert np.isfinite(performance).all(), (
        "Could not merge lists, because one list is empty, or the lists do not "
        "start with the same times and replace_nan is not set"
    )
    return performance, time_


def compute_mean_and_ci(
    metrics_runs: np.ndarray, time: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Aggregate is the mean, error bars are empirical estimate of 95% confidence
    interval for the true mean.

    Note: Error bar scale depends on number of runs `n` via `1 / sqrt(n)`.
    """
    mean = np.mean(metrics_runs.T, axis=0)
    std = np.std(metrics_runs.T, axis=0)
    num_runs = metrics_runs.shape[1]
    fact = 1.95 / np.sqrt(num_runs)
    return {
        "time": time,
        "aggregate": mean,
        "lower": mean - fact * std,
        "upper": mean + fact * std,
    }


def compute_median_percentiles(
    metrics_runs: np.ndarray, time: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Aggregate is the median, error bars are 25 and 75 percentiles.

    Note: Error bar scale does not depend on number of runs.
    """
    return {
        "time": time,
        "aggregate": np.median(metrics_runs.T, axis=0),
        "lower": np.percentile(metrics_runs.T, 25, axis=0),
        "upper": np.percentile(metrics_runs.T, 75, axis=0),
    }


def compute_iqm_bootstrap(
    metrics_runs: np.ndarray, time: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    The aggregate is the interquartile mean (IQM). Error bars are bootstrap
    estimate of 95% confidence interval for true IQM. This is the normal
    interval, based on the bootstrap variance estimate. While other bootstrap
    CI estimates are available, they are more expensive to compute.

    Note: Error bar scale depends on number of runs `n` via `1 / sqrt(n)`.
    """

    def _remove_mass(amat: np.ndarray, mass: float, col_index):
        remaining_mass = mass * np.ones(amat.shape[0])
        for col in col_index:
            subtract = np.minimum(remaining_mass, amat[:, col])
            remaining_mass -= subtract
            amat[:, col] -= subtract

    num_runs = metrics_runs.shape[1]
    # Sort matrix of metrics (in-place: `metrics_runs` overwritten)
    metrics_runs.sort(axis=1)
    # Interquartile mean
    indvec = np.ones((1, num_runs))
    _remove_mass(indvec, mass=num_runs / 4, col_index=range(num_runs))
    _remove_mass(indvec, mass=num_runs / 4, col_index=range(num_runs - 1, -1, -1))
    indvec *= 2 / num_runs
    iq_mean = np.dot(metrics_runs, indvec.reshape((-1, 1))).reshape((-1,))

    # Bootstrap variance estimates
    num_bootstrap_samples = 1000
    # Multinomial samples
    rng = default_rng()
    amat = rng.multinomial(
        n=num_runs, pvals=np.ones(num_runs) / num_runs, size=num_bootstrap_samples
    ).astype(np.float64)
    # Remove mass n/4 from left and right
    _remove_mass(amat, mass=num_runs / 4, col_index=range(num_runs))
    _remove_mass(amat, mass=num_runs / 4, col_index=range(num_runs - 1, -1, -1))
    amat *= 2 / num_runs
    # Bootstrap variances
    cvec = np.sum(amat, axis=0).reshape((-1, 1)) / num_bootstrap_samples
    cmat = np.dot(amat.T, amat) / num_bootstrap_samples
    tmpmat = np.dot(metrics_runs, cmat)
    svec = np.sum(tmpmat * metrics_runs, axis=1).reshape((-1,))
    mvec = np.dot(metrics_runs, cvec).reshape((-1,))
    bootstrap_vars = np.maximum(svec - np.square(mvec), 0)

    error = 1.96 * np.sqrt(bootstrap_vars)
    return {
        "time": time,
        "aggregate": iq_mean,
        "lower": iq_mean - error,
        "upper": iq_mean + error,
    }


def aggregate_and_errors_over_time(
    errors: List[np.ndarray], runtimes: List[np.ndarray], mode: str = "mean_and_ci"
) -> Dict[str, np.ndarray]:
    min_t = np.max([runtime[0] for runtime in runtimes])
    metrics_runs, time = fill_trajectory(errors, runtimes, replace_nan=1)
    idx = time.tolist().index(min_t)
    metrics_runs = metrics_runs[idx:, :]
    time = time[idx:]
    if mode == "mean_and_ci":
        return compute_mean_and_ci(metrics_runs, time)
    elif mode == "median_percentiles":
        return compute_median_percentiles(metrics_runs, time)
    elif mode == "iqm_bootstrap":
        return compute_iqm_bootstrap(metrics_runs, time)
    else:
        raise ValueError(f"mode = {mode} not supported")
