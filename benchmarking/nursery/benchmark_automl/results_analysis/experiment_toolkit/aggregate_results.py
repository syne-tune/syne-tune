import numpy as np
import pandas as pd
from numpy.random import default_rng

from collections import OrderedDict


def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))

        ds = pd.Series(data=p, index=t)
        ds = ds[~ds.index.duplicated(keep='first')]
        frame_dict[str(c)] = ds
    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()

    performance = merged.values
    time_ = merged.index.values

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


def average_performance_over_time(error, runtimes):
    t = np.max([runtimes[i][0] for i in range(len(runtimes))])

    te, time = fill_trajectory(error, runtimes, replace_nan=1)

    idx = time.tolist().index(t)
    te = te[idx:, :]
    time = time[idx:]
    mean = np.mean(te.T, axis=0)
    std = np.std(te.T, axis=0)

    return time, mean, std


def median_performance_over_time(error, runtimes, return_quantile=False):
    t = np.max([runtimes[i][0] for i in range(len(runtimes))])

    te, time = fill_trajectory(error, runtimes, replace_nan=1)

    idx = time.tolist().index(t)
    te = te[idx:, :]
    time = time[idx:]
    median = np.median(te.T, axis=0)

    if return_quantile:
        return time, median, np.percentile(te.T, 25, axis=0), np.percentile(te.T, 75, axis=0)
    return time, median


def compute_regret(df, groupby, objective_name, runtime_name, y_opt, aggregate='mean'):
    traj = []
    runtime = []
    for exp_name, run in df.groupby(groupby):
        traj.append(np.array(run[objective_name].cummin()) - y_opt)
        runtime.append(np.array(run[runtime_name]))

    if aggregate == 'mean':
        x, y, std = average_performance_over_time(error=traj,
                                                  runtimes=runtime)
        return x, y, y + std, y - std

    else:
        x, y, p25, p75 = median_performance_over_time(error=traj,
                                                      runtimes=runtime,
                                                      return_quantile=True)

        return x, y, p25, p75


def _compute_mean_and_ci(metrics_runs, time) -> dict:
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
        'time': time,
        'aggregate': mean,
        'lower': mean - fact * std,
        'upper': mean + fact * std,
    }


def _compute_median_percentiles(metrics_runs, time) -> dict:
    """
    Aggregate is the median, error bars are 25 and 75 percentiles.

    Note: Error bar scale does not depend on number of runs.
    """
    return {
        'time': time,
        'aggregate': np.median(metrics_runs.T, axis=0),
        'lower': np.percentile(metrics_runs.T, 25, axis=0),
        'upper': np.percentile(metrics_runs.T, 75, axis=0),
    }


def _remove_mass(amat, mass, col_index):
    remaining_mass = mass * np.ones(amat.shape[0])
    for col in col_index:
        subtract = np.minimum(remaining_mass, amat[:, col])
        remaining_mass -= subtract
        amat[:, col] -= subtract


def _compute_iqm_bootstrap(metrics_runs, time) -> dict:
    """
    The aggregate is the interquartile mean (IQM). Error bars are bootstrap
    estimate of 95% confidence interval for true IQM. This is the normal
    interval, based on the bootstrap variance estimate. While other bootstrap
    CI estimates are available, they are more expensive to compute.

    Note: Error bar scale depends on number of runs `n` via `1 / sqrt(n)`.

    """
    num_runs = metrics_runs.shape[1]
    # Sort matrix of metrics (in-place: `metrics_runs` overwritten)
    metrics_runs.sort(axis=1)
    # Interquartile mean
    indvec = np.ones((1, num_runs))
    _remove_mass(indvec, mass=num_runs / 4, col_index=range(num_runs))
    _remove_mass(indvec, mass=num_runs / 4,
                 col_index=range(num_runs - 1, -1, -1))
    indvec *= (2 / num_runs)
    iq_mean = np.dot(metrics_runs, indvec.reshape((-1, 1))).reshape((-1,))

    # Bootstrap variance estimates
    num_bootstrap_samples = 1000
    # Multinomial samples
    rng = default_rng()
    amat = rng.multinomial(
        n=num_runs, pvals=np.ones(num_runs) / num_runs,
        size=num_bootstrap_samples).astype(np.float64)
    # Remove mass n/4 from left and right
    _remove_mass(amat, mass=num_runs / 4, col_index=range(num_runs))
    _remove_mass(amat, mass=num_runs / 4,
                 col_index=range(num_runs - 1, -1, -1))
    amat *= (2 / num_runs)
    # Bootstrap variances
    cvec = np.sum(amat, axis=0).reshape((-1, 1)) / num_bootstrap_samples
    cmat = np.dot(amat.T, amat) / num_bootstrap_samples
    tmpmat = np.dot(metrics_runs, cmat)
    svec = np.sum(tmpmat * metrics_runs, axis=1).reshape((-1,))
    mvec = np.dot(metrics_runs, cvec).reshape((-1,))
    bootstrap_vars = np.maximum(svec - np.square(mvec), 0)

    error = 1.96 * np.sqrt(bootstrap_vars)
    return {
        'time': time,
        'aggregate': iq_mean,
        'lower': iq_mean - error,
        'upper': iq_mean + error,
    }


def aggregate_and_errors_over_time(
        errors, runtimes, mode='mean_and_ci') -> dict:
    min_t = np.max([runtimes[i][0] for i in range(len(runtimes))])
    metrics_runs, time = fill_trajectory(errors, runtimes, replace_nan=1)
    idx = time.tolist().index(min_t)
    metrics_runs = metrics_runs[idx:, :]
    time = time[idx:]
    if mode == 'mean_and_ci':
        return _compute_mean_and_ci(metrics_runs, time)
    elif mode == 'median_percentiles':
        return _compute_median_percentiles(metrics_runs, time)
    elif mode == 'iqm_bootstrap':
        return _compute_iqm_bootstrap(metrics_runs, time)
    else:
        raise ValueError(f"mode = {mode} not supported")
