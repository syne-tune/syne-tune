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
from typing import List, Dict, Optional, Tuple

import numpy as np
import scipy.linalg as spl
import autograd.numpy as anp
from autograd.scipy.special import logsumexp
from autograd.scipy.linalg import solve_triangular
from autograd.tracer import getval
from numpy.random import RandomState
from operator import itemgetter
from collections import Counter


from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    NUMERICAL_JITTER,
    MIN_POSTERIOR_VARIANCE,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.custom_op import (
    cholesky_factorization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    FantasizedPendingEvaluation,
    TrialEvaluations,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext import (
    ExtendedConfiguration,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler


def _prepare_data_internal(
    state: TuningJobState,
    data_lst: List[Tuple[Configuration, List, str]],
    config_space_ext: ExtendedConfiguration,
    active_metric: str,
    do_fantasizing: bool,
    mean: float,
    std: float,
) -> (List[Configuration], List[np.ndarray], List[str]):
    r_min, r_max = config_space_ext.resource_attr_range
    configs = [x[0] for x in data_lst]
    trial_ids = [x[2] for x in data_lst]
    targets = []

    fantasized = dict()
    num_fantasy_samples = None
    if do_fantasizing:
        for ev in state.pending_evaluations:
            assert isinstance(ev, FantasizedPendingEvaluation)
            trial_id = ev.trial_id
            entry = (ev.resource, ev.fantasies[active_metric])
            sz = entry[1].size
            if num_fantasy_samples is None:
                num_fantasy_samples = sz
            else:
                assert sz == num_fantasy_samples, (
                    "Number of fantasy samples must be the same for all "
                    + f"pending evaluations ({sz}, {num_fantasy_samples})"
                )
            if trial_id in fantasized:
                fantasized[trial_id].append(entry)
            else:
                fantasized[trial_id] = [entry]

    trial_ids_done = set()
    for config, observed, trial_id in data_lst:
        # Observations must be from r_min without any missing
        obs_res = [x[0] for x in observed]
        num_obs = len(observed)
        if num_obs > 0:
            test = list(range(r_min, r_min + num_obs))
            assert obs_res == test, (
                f"trial_id {trial_id} has observations at {obs_res}, but "
                + f"we need them at {test}"
            )
        # Note: Only observed targets are normalized, not fantasized ones
        this_targets = (
            np.array([x[1] for x in observed]).reshape((-1, 1)) - mean
        ) / std
        if do_fantasizing:
            if num_fantasy_samples > 1:
                this_targets = this_targets * np.ones((1, num_fantasy_samples))
            if trial_id in fantasized:
                this_fantasized = sorted(fantasized[trial_id], key=itemgetter(0))
                fanta_res = [x[0] for x in this_fantasized]
                start = r_min + num_obs
                test = list(range(start, start + len(this_fantasized)))
                assert fanta_res == test, (
                    f"trial_id {trial_id} has pending evaluations at {fanta_res}"
                    + f", but we need them at {test}"
                )
                this_targets = np.vstack(
                    [this_targets] + [x[1].reshape((1, -1)) for x in this_fantasized]
                )
                trial_ids_done.add(trial_id)
        targets.append(this_targets)

    if do_fantasizing:
        # There may be trials with pending evals, but no observes ones
        for trial_id, this_fantasized in fantasized.items():
            if trial_id not in trial_ids_done:
                configs.append(state.config_for_trial[trial_id])
                trial_ids.append(trial_id)
                this_fantasized = sorted(this_fantasized, key=itemgetter(0))
                fanta_res = [x[0] for x in this_fantasized]
                test = list(range(r_min, r_min + len(this_fantasized)))
                assert fanta_res == test, (
                    f"trial_id {trial_id} has pending evaluations at {fanta_res}"
                    + f", but we need them at {test}"
                )
                this_targets = np.vstack(
                    [x[1].reshape((1, -1)) for x in this_fantasized]
                )
                targets.append(this_targets)

    return configs, targets, trial_ids


def _create_tuple(ev: TrialEvaluations, active_metric: str, config_for_trial: Dict):
    metric_vals = ev.metrics[active_metric]
    assert isinstance(metric_vals, dict)
    observed = list(
        sorted(((int(k), v) for k, v in metric_vals.items()), key=itemgetter(0))
    )
    trial_id = ev.trial_id
    config = config_for_trial[trial_id]
    return config, observed, trial_id


def prepare_data(
    state: TuningJobState,
    config_space_ext: ExtendedConfiguration,
    active_metric: str,
    normalize_targets: bool = False,
    do_fantasizing: bool = False,
) -> Dict:
    """
    Prepares data in `state` for further processing. The entries
    `configs`, `targets` of the result dict are lists of one entry per trial,
    they are sorted in decreasing order of number of target values. `features`
    is the feature matrix corresponding to `configs`. If `normalize_targets`
    is True, the target values are normalized to mean 0, variance 1 (over all
    values), and `mean_targets`, `std_targets` is returned.

    If `do_fantasizing` is True, `state.pending_evaluations` is also taken into
    account. Entries there have to be of type `FantasizedPendingEvaluation`.
    Also, in terms of their resource levels, they need to be adjacent to
    observed entries, so there are no gaps. In this case, the entries of the
    `targets` list are matrices, each column corr´esponding to a fantasy sample.

    Note: If `normalize_targets`, mean and stddev are computed over observed
    values only. Also, fantasy values in `state.pending_evaluations` are not
    normalized, because they are assumed to be sampled from the posterior with
    normalized targets as well.

    :param state: `TuningJobState` with data
    :param config_space_ext: Extended config space
    :param active_metric:
    :param normalize_targets: See above
    :param do_fantasizing: See above
    :return: See above
    """
    r_min, r_max = config_space_ext.resource_attr_range
    hp_ranges = config_space_ext.hp_ranges
    data_lst = []
    targets = []
    for ev in state.trials_evaluations:
        tpl = _create_tuple(ev, active_metric, state.config_for_trial)
        data_lst.append(tpl)
        observed = tpl[1]
        targets += [x[1] for x in observed]
    mean = 0.0
    std = 1.0
    if normalize_targets:
        std = max(np.std(targets), 1e-9)
        mean = np.mean(targets)

    configs, targets, trial_ids = _prepare_data_internal(
        state=state,
        data_lst=data_lst,
        config_space_ext=config_space_ext,
        active_metric=active_metric,
        do_fantasizing=do_fantasizing,
        mean=mean,
        std=std,
    )
    # Sort in decreasing order w.r.t. number of targets
    configs, targets, trial_ids = zip(
        *sorted(zip(configs, targets, trial_ids), key=lambda x: -x[1].shape[0])
    )
    features = hp_ranges.to_ndarray_matrix(configs)
    result = {
        "configs": list(configs),
        "features": features,
        "targets": list(targets),
        "trial_ids": list(trial_ids),
        "r_min": r_min,
        "r_max": r_max,
        "do_fantasizing": do_fantasizing,
    }
    if normalize_targets:
        result["mean_targets"] = mean
        result["std_targets"] = std
    return result


def prepare_data_with_pending(
    state: TuningJobState,
    config_space_ext: ExtendedConfiguration,
    active_metric: str,
    normalize_targets: bool = False,
) -> (Dict, Dict):
    """
    Similar to `prepare_data` with `do_fantasizing=False`, but two dicts are
    returned, the first for trials without pending evaluations, the second
    for trials with pending evaluations. The latter dict also contains trials
    which have pending, but no observed evaluations.
    The second dict has the additional entry `num_pending`, which lists the
    number of pending evals for each trial. These evals must be contiguous and
    adjacent with observed evals, so that the union of observed and pending
    evals are contiguous (when it comes to resource levels).

    :param state: See `prepare_data`
    :param config_space_ext: See `prepare_data`
    :param active_metric: See `prepare_data`
    :param normalize_targets: See `prepare_data`
    :return: See above

    """
    r_min, r_max = config_space_ext.resource_attr_range
    hp_ranges = config_space_ext.hp_ranges
    data1_lst = []  # trials without pending evals
    data2_lst = []  # trials with pending evals
    num_pending = []
    num_pending_for_trial = Counter(ev.trial_id for ev in state.pending_evaluations)
    targets = []
    done_trial_ids = set()
    for ev in state.trials_evaluations:
        tpl = _create_tuple(ev, active_metric, state.config_for_trial)
        _, observed, trial_id = tpl
        if trial_id not in num_pending_for_trial:
            data1_lst.append(tpl)
        else:
            data2_lst.append(tpl)
            num_pending.append(num_pending_for_trial[trial_id])
        done_trial_ids.add(trial_id)
        targets += [x[1] for x in observed]
    mean = 0.0
    std = 1.0
    if normalize_targets:
        std = max(np.std(targets), 1e-9)
        mean = np.mean(targets)
    # There may be trials with pending evaluations, but no observed ones
    for ev in state.pending_evaluations:
        trial_id = ev.trial_id
        if trial_id not in done_trial_ids:
            config = state.config_for_trial[trial_id]
            data2_lst.append((config, [], trial_id))
            num_pending.append(num_pending_for_trial[trial_id])

    results = ()
    with_pending = False
    for data_lst in (data1_lst, data2_lst):
        configs, targets, trial_ids = _prepare_data_internal(
            state=state,
            data_lst=data_lst,
            config_space_ext=config_space_ext,
            active_metric=active_metric,
            do_fantasizing=False,
            mean=mean,
            std=std,
        )
        if configs:
            # Sort in decreasing order w.r.t. number of targets
            if not with_pending:
                configs, targets, trial_ids = zip(
                    *sorted(
                        zip(configs, targets, trial_ids), key=lambda x: -x[1].shape[0]
                    )
                )
            else:
                configs, targets, num_pending, trial_ids = zip(
                    *sorted(
                        zip(configs, targets, num_pending, trial_ids),
                        key=lambda x: -x[1].shape[0],
                    )
                )
            features = hp_ranges.to_ndarray_matrix(configs)
        else:
            # It is possible that `data1_lst` is empty
            features = None
        result = {
            "configs": list(configs),
            "features": features,
            "targets": list(targets),
            "trial_ids": list(trial_ids),
            "r_min": r_min,
            "r_max": r_max,
            "do_fantasizing": False,
        }
        if with_pending:
            result["num_pending"] = num_pending
        if normalize_targets:
            result["mean_targets"] = mean
            result["std_targets"] = std
        results = results + (result,)
        with_pending = True
    return results


def issm_likelihood_precomputations(targets: List[np.ndarray], r_min: int) -> Dict:
    """
    Precomputations required by `issm_likelihood_computations`.

    Importantly, `prepare_data` orders datapoints by nonincreasing number of
    targets `ydims[i]`. For `0 <= j < ydim_max`, `ydim_max = ydims[0] =
    max(ydims)`, `num_configs[j]` is the number of datapoints i for which
    `ydims[i] > j`.
    `deltay` is a flat matrix (rows corresponding to fantasy samples; column
    vector if no fantasizing) consisting of `ydim_max` parts, where part j is
    of size `num_configs[j]` and contains `y[j] - y[j-1]` for targets of
    those i counted in `num_configs[j]`, the term needed in the recurrence to
    compute `w[j]`.
    'logr` is a flat vector consisting of `ydim_max - 1` parts, where part j
    (starting from 1) is of size `num_configs[j]` and contains the logarithmic
    term for computing `a[j-1]` and `e[j]`.

    :param targets: Targets from data representation returned by
        `prepare_data`
    :param r_min: Value of r_min, as returned by `prepare_data`
    :return: See above
    """
    ydims = [y.shape[0] for y in targets]
    ydim_max = ydims[0]
    num_configs = list(
        np.sum(
            np.array(ydims).reshape((-1, 1)) > np.arange(ydim_max).reshape((1, -1)),
            axis=0,
        ).reshape((-1,))
    )
    assert num_configs[0] == len(targets), (num_configs, len(targets))
    assert num_configs[-1] > 0, num_configs
    total_size = sum(num_configs)
    assert total_size == sum(ydims)
    yprev = np.vstack([y[-1].reshape((1, -1)) for y in targets])
    deltay_parts = [yprev]
    log_r = []
    for pos, num in enumerate(num_configs[1:], start=1):
        ycurr = np.vstack([y[-(pos + 1)].reshape((1, -1)) for y in targets[:num]])
        deltay_parts.append(ycurr - yprev[:num, :])
        yprev = ycurr
        logr_curr = [np.log(ydim + r_min - pos) for ydim in ydims[:num]]
        log_r.extend(logr_curr)
    deltay = np.vstack(deltay_parts)
    assert deltay.shape[0] == total_size
    assert len(log_r) == total_size - num_configs[0]
    return {
        "ydims": ydims,
        "num_configs": num_configs,
        "deltay": deltay,
        "logr": np.array(log_r),
    }


def _squared_norm(a, _np=anp):
    return _np.sum(_np.square(a))


def _inner_product(a, b, _np=anp):
    return _np.sum(_np.multiply(a, b))


def _colvec(a, _np=anp):
    return _np.reshape(a, (-1, 1))


def _rowvec(a, _np=anp):
    return _np.reshape(a, (1, -1))


def _flatvec(a, _np=anp):
    return _np.reshape(a, (-1,))


def issm_likelihood_computations(
    precomputed: Dict,
    issm_params: Dict,
    r_min: int,
    r_max: int,
    skip_c_d: bool = False,
    profiler: Optional[SimpleProfiler] = None,
) -> Dict:
    """
    Given `precomputed` from `issm_likelihood_precomputations` and ISSM
    parameters `issm_params`, compute quantities required for inference and
    marginal likelihood computation, pertaining to the ISSM likelihood.

    The index for r is range(r_min, r_max + 1). Observations must be contiguous
    from r_min. The ISSM parameters are:
    - alpha: n-vector, negative
    - beta: n-vector
    - gamma: scalar, positive

    Results returned are:
    - c: n vector [c_i], negative
    - d: n vector [d_i], positive
    - vtv: n vector [|v_i|^2]
    - wtv: (n, F) matrix [(W_i)^T v_i], F number of fantasy samples
    - wtw: n-vector [|w_i|^2] (only if no fantasizing)

    :param precomputed: Output of `issm_likelihood_precomputations`
    :param issm_params: Parameters of ISSM likelihood
    :param r_min: Smallest resource value
    :param r_max: Largest resource value
    :param skip_c_d: If True, c and d are not computed
    :return: Quantities required for inference and learning criterion

    """
    num_all_configs = precomputed["num_configs"][0]
    num_res = r_max + 1 - r_min
    assert num_all_configs > 0, "targets must not be empty"
    assert num_res > 0, f"r_min = {r_min} must be <= r_max = {r_max}"
    num_fantasy_samples = precomputed["deltay"].shape[1]
    compute_wtw = num_fantasy_samples == 1
    alphas = _flatvec(issm_params["alpha"])
    betas = _flatvec(issm_params["beta"])
    gamma = issm_params["gamma"]
    n = getval(alphas.size)
    assert n == num_all_configs, f"alpha.size = {n} != {num_all_configs}"
    n = getval(betas.size)
    assert n == num_all_configs, f"beta.size = {n} != {num_all_configs}"

    if not skip_c_d:
        # We could probably refactor this to fit into the loop below, but it
        # seems subdominant
        if profiler is not None:
            profiler.start("issm_part1")
        c_lst = []
        d_lst = []
        for i, ydim in enumerate(precomputed["ydims"]):
            alpha = alphas[i]
            alpha_m1 = alpha - 1.0
            beta = betas[i]
            r_obs = r_min + ydim  # Observed in range(r_min, r_obs)
            assert 0 < ydim <= num_res, f"len(y[{i}]) = {ydim}, num_res = {num_res}"
            # c_i, d_i
            if ydim < num_res:
                lrvec = (
                    anp.array([np.log(r) for r in range(r_obs, r_max + 1)]) * alpha_m1
                    + beta
                )
                c_scal = alpha * anp.exp(logsumexp(lrvec))
                d_scal = anp.square(gamma * alpha) * anp.exp(logsumexp(lrvec * 2.0))
                c_lst.append(c_scal)
                d_lst.append(d_scal)
            else:
                c_lst.append(0.0)
                d_lst.append(0.0)
        if profiler is not None:
            profiler.stop("issm_part1")

    # Loop over ydim
    if profiler is not None:
        profiler.start("issm_part2")
    deltay = precomputed["deltay"]
    logr = precomputed["logr"]
    off_dely = num_all_configs
    vvec = anp.ones(off_dely)
    wmat = deltay[:off_dely, :]  # [y_0]
    vtv = anp.ones(off_dely)
    wtv = wmat.copy()
    if compute_wtw:
        wtw = _flatvec(anp.square(wmat))
    # Note: We need the detour via `vtv_lst`, etc, because `autograd` does not
    # support overwriting the content of an `ndarray`. Their role is to collect
    # parts of the final vectors, in reverse ordering
    vtv_lst = []
    wtv_lst = []
    wtw_lst = []
    alpham1s = alphas - 1
    num_prev = off_dely
    for num in precomputed["num_configs"][1:]:
        if num < num_prev:
            # Size of working vectors is shrinking
            assert vtv.size == num_prev
            # These parts are done: Collect them in the lists
            # All vectors are resized to `num`, dropping the tails
            vtv_lst.append(vtv[num:])
            wtv_lst.append(wtv[num:, :])
            vtv = vtv[:num]
            wtv = wtv[:num, :]
            if compute_wtw:
                wtw_lst.append(wtw[num:])
                wtw = wtw[:num]
            alphas = alphas[:num]
            alpham1s = alpham1s[:num]
            betas = betas[:num]
            vvec = vvec[:num]
            wmat = wmat[:num, :]
            num_prev = num
        # [a_{j-1}]
        off_logr = off_dely - num_all_configs
        logr_curr = logr[off_logr : (off_logr + num)]
        avec = alphas * anp.exp(logr_curr * alpham1s + betas)
        evec = avec * gamma + 1  # [e_j]
        vvec = vvec * evec  # [v_j]
        deltay_curr = deltay[off_dely : (off_dely + num), :]
        off_dely += num
        wmat = _colvec(evec) * wmat + deltay_curr + _colvec(avec)  # [w_j]
        vtv = vtv + anp.square(vvec)
        if compute_wtw:
            wtw = wtw + _flatvec(anp.square(wmat))
        wtv = wtv + _colvec(vvec) * wmat
    vtv_lst.append(vtv)
    wtv_lst.append(wtv)
    vtv_all = anp.concatenate(tuple(reversed(vtv_lst)), axis=0)
    wtv_all = anp.concatenate(tuple(reversed(wtv_lst)), axis=0)
    if compute_wtw:
        wtw_lst.append(wtw)
        wtw_all = anp.concatenate(tuple(reversed(wtw_lst)), axis=0)
    if profiler is not None:
        profiler.stop("issm_part2")

    # Compile results
    result = {"num_data": sum(precomputed["ydims"]), "vtv": vtv_all, "wtv": wtv_all}
    if compute_wtw:
        result["wtw"] = wtw_all
    if not skip_c_d:
        result["c"] = anp.array(c_lst)
        result["d"] = anp.array(d_lst)
    return result


def posterior_computations(
    features, mean, kernel, issm_likelihood: Dict, noise_variance
) -> Dict:
    """
    Computes posterior state (required for predictions) and negative log
    marginal likelihood (returned in `criterion`), The latter is computed only
    when there is no fantasizing (i.e., if `issm_likelihood` contains `wtw`).

    :param features: Input matrix X
    :param mean: Mean function
    :param kernel: Kernel function
    :param issm_likelihood: Outcome of `issm_likelihood_computations`
    :param noise_variance: Variance of ISSM innovations
    :return: Internal posterior state

    """
    num_data = issm_likelihood["num_data"]
    kernel_mat = kernel(features, features)  # K
    dvec = issm_likelihood["d"]
    s2vec = issm_likelihood["vtv"] / noise_variance  # S^2
    svec = _colvec(anp.sqrt(s2vec + NUMERICAL_JITTER))
    # A = I + S K_hat S = (I + S^2 D) + S K S
    dgvec = s2vec * dvec + 1.0
    amat = anp.multiply(svec, anp.multiply(kernel_mat, _rowvec(svec))) + anp.diag(dgvec)
    # TODO: Do we need AddJitterOp here?
    lfact = cholesky_factorization(amat)  # L (Cholesky factor)
    # r vectors
    muhat = _flatvec(mean(features)) - issm_likelihood["c"]
    s2muhat = s2vec * muhat  # S^2 mu_hat
    r2mat = issm_likelihood["wtv"] / noise_variance - _colvec(s2muhat)
    r3mat = anp.matmul(kernel_mat, r2mat) + _colvec(dvec) * r2mat
    r4mat = solve_triangular(lfact, r3mat * svec, lower=True)
    # Prediction matrix P
    pmat = r2mat - svec * solve_triangular(lfact, r4mat, lower=True, trans="T")
    result = {
        "features": features,
        "chol_fact": lfact,
        "svec": _flatvec(svec),
        "pmat": pmat,
        "likelihood": issm_likelihood,
    }
    if "wtw" in issm_likelihood:
        # Negative log marginal likelihood
        # Part sigma^{-2} |r_1|^2 - (r_2)^T r_3 + |r_4|^2
        r2vec = _flatvec(r2mat)
        r3vec = _flatvec(r3mat)
        r4vec = _flatvec(r4mat)
        # We use: r_1 = w - V mu_hat, sigma^{-2} V^T V = S^2, and
        # r_2 = sigma^{-2} V^T w - S^2 mu_hat, so that:
        # sigma^{-2} |r_1|^2
        # = sigma^{-2} |w|^2 + |S mu_hat|^2 - 2 sigma^{-2} w^T V mu_hat
        # = sigma^{-2} |w|^2 - |S mu_hat|^2 - 2 * (r_2)^T mu_hat
        # = sigma^{-2} |w|^2 - mu_hat^T (S^2 mu_hat + 2 r_2)
        part2 = 0.5 * (
            anp.sum(issm_likelihood["wtw"]) / noise_variance
            - _inner_product(muhat, s2muhat + 2.0 * r2vec)
            - _inner_product(r2vec, r3vec)
            + _squared_norm(r4vec)
        )
        part1 = anp.sum(anp.log(anp.abs(anp.diag(lfact)))) + 0.5 * num_data * anp.log(
            2 * anp.pi * noise_variance
        )
        result["criterion"] = part1 + part2
        result["r2vec"] = r2vec
        result["r4vec"] = r4vec
    return result


def predict_posterior_marginals(poster_state: Dict, mean, kernel, test_features):
    """
    These are posterior marginals on the h variable, whereas the full model is
    for f_r = h + g_r (additive).
    `posterior_means` is a (n, F) matrix, where F is the number of fantasy
    samples, or F == 1 without fantasizing.

    :param poster_state: Posterior state
    :param mean: Mean function
    :param kernel: Kernel function
    :param test_features: Feature matrix for test points (not extended)
    :return: posterior_means, posterior_variances
    """
    k_tr_te = kernel(poster_state["features"], test_features)
    posterior_means = anp.matmul(k_tr_te.T, poster_state["pmat"]) + _colvec(
        mean(test_features)
    )
    qmat = solve_triangular(
        poster_state["chol_fact"],
        anp.multiply(_colvec(poster_state["svec"]), k_tr_te),
        lower=True,
    )
    posterior_variances = kernel.diagonal(test_features) - anp.sum(
        anp.square(qmat), axis=0
    )
    return posterior_means, _flatvec(
        anp.maximum(posterior_variances, MIN_POSTERIOR_VARIANCE)
    )


def sample_posterior_marginals(
    poster_state: Dict,
    mean,
    kernel,
    test_features,
    random_state: RandomState,
    num_samples: int = 1,
):
    """
    We sample from posterior marginals on the h variance, see also
    `predict_posterior_marginals`.
    """
    post_means, post_vars = predict_posterior_marginals(
        poster_state, mean, kernel, test_features
    )
    assert getval(post_means.shape[1]) == 1, (
        "sample_posterior_marginals cannot be used for posterior state "
        + "based on fantasizing"
    )
    n01_mat = random_state.normal(size=(getval(post_means.shape[0]), num_samples))
    post_stds = _colvec(anp.sqrt(post_vars))
    return anp.multiply(post_stds, n01_mat) + _colvec(post_means)


def predict_posterior_marginals_extended(
    poster_state: Dict,
    mean,
    kernel,
    test_features,
    resources: List[int],
    issm_params: Dict,
    r_min: int,
    r_max: int,
):
    """
    These are posterior marginals on f_r = h + g_r variables, where
    (x, r) are zipped from `test_features`, `resources`. `issm_params`
    are likelihood parameters for the test configs.
    `posterior_means` is a (n, F) matrix, where F is the number of fantasy
    samples, or F == 1 without fantasizing.

    :param poster_state: Posterior state
    :param mean: Mean function
    :param kernel: Kernel function
    :param test_features: Feature matrix for test points (not extended)
    :param resources: Resource values corresponding to rows of
        `test_features`
    :param issm_params: See above
    :param r_min:
    :param r_max:
    :return: posterior_means, posterior_variances

    """
    num_test = test_features.shape[0]
    assert len(resources) == num_test, (
        f"test_features.shape[0] = {num_test} != {len(resources)} " + "= len(resources)"
    )
    alphas = anp.reshape(issm_params["alpha"], (-1,))
    betas = anp.reshape(issm_params["beta"], (-1,))
    gamma = issm_params["gamma"]
    n = getval(alphas.size)
    assert n == num_test, (
        f"Entries in issm_params must have size {num_test}, but " + f"have size {n}"
    )
    # Predictive marginals over h
    h_means, h_variances = predict_posterior_marginals(
        poster_state, mean, kernel, test_features
    )
    if all(r == r_max for r in resources):
        # Frequent special case
        posterior_means = h_means
        posterior_variances = h_variances
    else:
        # Convert into predictive marginals over f_r
        posterior_means = []
        posterior_variances = []
        for h_mean, h_variance, resource, alpha, beta in zip(
            h_means, h_variances, resources, alphas, betas
        ):
            sz = r_max - resource
            h_mean = _rowvec(h_mean)
            if sz == 0:
                posterior_means.append(h_mean)
                posterior_variances.append(h_variance)
            else:
                lrvec = (
                    anp.array([np.log(r_max - t) for t in range(sz)]) * (alpha - 1.0)
                    + beta
                )
                avec = alpha * anp.exp(lrvec)
                a2vec = anp.square(alpha * gamma) * anp.exp(lrvec * 2.0)
                c = anp.sum(avec)
                d = anp.sum(a2vec)
                posterior_means.append(h_mean - c)
                posterior_variances.append(h_variance + d)
        posterior_means = anp.vstack(posterior_means)
        posterior_variances = anp.array(posterior_variances)
    return posterior_means, posterior_variances


def sample_posterior_joint(
    poster_state: Dict,
    mean,
    kernel,
    feature,
    targets: np.ndarray,
    issm_params: Dict,
    r_min: int,
    r_max: int,
    random_state: RandomState,
    num_samples: int = 1,
) -> Dict:
    """
    Given `poster_state` for some data plus one additional configuration
    with data (`feature`, `targets`, `issm_params`), draw joint samples
    of the latent variables not fixed by the data, and of the latent
    target values. `targets` may be empty, but must not reach all the
    way to `r_max`. The additional configuration must not be in the
    dataset used to compute `poster_state`.

    If `targets` correspond to resource values range(r_min, r_obs), we
    sample latent target values y_r corresponding to range(r_obs, r_max+1)
    and latent function values f_r corresponding to range(r_obs-1, r_max+1),
    unless r_obs = r_min (i.e. `targets` empty), in which case both [y_r]
    and [f_r] ranges in range(r_min, r_max+1). We return a dict with
    [f_r] under `f`, [y_r] under `y`. These are matrices with `num_samples`
    columns.

    :param poster_state: Posterior state for data
    :param mean: Mean function
    :param kernel: Kernel function
    :param feature: Features for additional config
    :param targets: Target values for additional config
    :param issm_params: Likelihood parameters for additional config
    :param r_min: Smallest resource value
    :param r_max: Largest resource value
    :param random_state: numpy.random.RandomState
    :param num_samples: Number of joint samples to draw (default: 1)
    :return: See above
    """
    num_res = r_max + 1 - r_min
    targets = _colvec(targets, _np=np)
    ydim = targets.size
    t_obs = num_res - ydim
    assert t_obs > 0, f"targets.size = {ydim} must be < {num_res}"
    assert getval(poster_state["pmat"].shape[1]) == 1, (
        "sample_posterior_joint cannot be used for posterior state "
        + "based on fantasizing"
    )
    # ISSM parameters
    alpha = issm_params["alpha"][0]
    alpha_m1 = alpha - 1.0
    beta = issm_params["beta"][0]
    gamma = issm_params["gamma"]
    # Posterior mean and variance of h for additional config
    post_mean, post_variance = predict_posterior_marginals(
        poster_state, mean, kernel, _rowvec(feature, _np=np)
    )
    post_mean = post_mean[0].item()
    post_variance = post_variance[0].item()
    # Compute [a_t], [gamma^2 a_t^2]
    lrvec = np.array([np.log(r_max - t) for t in range(num_res - 1)]) * alpha_m1 + beta
    avec = alpha * np.exp(lrvec)
    a2vec = np.square(alpha * gamma) * np.exp(lrvec * 2.0)
    # Draw the [eps_t] for all samples
    epsmat = random_state.normal(size=(num_res, num_samples))
    # Compute samples [f_t], [y_t], not conditioned on targets
    hvec = (
        random_state.normal(size=(1, num_samples)) * np.sqrt(post_variance) + post_mean
    )
    f_rows = []
    y_rows = []
    fcurr = hvec
    for t in range(num_res - 1):
        eps_row = _rowvec(epsmat[t], _np=np)
        f_rows.append(fcurr)
        y_rows.append(fcurr + eps_row)
        fcurr = fcurr - avec[t] * (eps_row * gamma + 1.0)
    eps_row = _rowvec(epsmat[-1], _np=np)
    f_rows.append(fcurr)
    y_rows.append(fcurr + eps_row)
    if ydim > 0:
        # Condition on targets
        # Prior samples (reverse order t -> r)
        fsamples = np.concatenate(tuple(reversed(f_rows[: (t_obs + 1)])), axis=0)
        # Compute c1 and d1 vectors (same for all samples)
        zeroscal = np.zeros((1,))
        c1vec = np.flip(np.concatenate((zeroscal, np.cumsum(avec[:t_obs])), axis=None))
        d1vec = np.flip(np.concatenate((zeroscal, np.cumsum(a2vec[:t_obs])), axis=None))
        # Assemble targets for conditional means
        ymat = np.concatenate(tuple(reversed(y_rows[t_obs:])), axis=0)
        ycols = np.split(ymat, num_samples, axis=1)
        assert ycols[0].size == ydim  # Sanity check
        # v^T v, w^T v for sampled targets
        onevec = np.ones((num_samples,))
        _issm_params = {"alpha": alpha * onevec, "beta": beta * onevec, "gamma": gamma}
        issm_likelihood = issm_likelihood_slow_computations(
            targets=[_colvec(v, _np=np) for v in ycols],
            issm_params=_issm_params,
            r_min=r_min,
            r_max=r_max,
            skip_c_d=True,
        )
        vtv = issm_likelihood["vtv"]
        wtv = issm_likelihood["wtv"]
        # v^T v, w^T v for observed (last entry)
        issm_likelihood = issm_likelihood_slow_computations(
            targets=[targets], issm_params=issm_params, r_min=r_min, r_max=r_max
        )
        vtv = _rowvec(np.concatenate((vtv, issm_likelihood["vtv"]), axis=None), _np=np)
        wtv = _rowvec(np.concatenate((wtv, issm_likelihood["wtv"]), axis=None), _np=np)
        cscal = issm_likelihood["c"][0]
        dscal = issm_likelihood["d"][0]
        c1vec = _colvec(c1vec, _np=np)
        d1vec = _colvec(d1vec, _np=np)
        c2vec = cscal - c1vec
        d2vec = dscal - d1vec
        # Compute num_samples + 1 conditional mean vectors in one go
        denom = vtv * (post_variance + dscal) + 1.0
        cond_means = (
            (post_mean - c1vec) * (d2vec * vtv + 1.0)
            + (d1vec + post_variance) * (c2vec * vtv + wtv)
        ) / denom
        fsamples = (
            fsamples
            - cond_means[:, :num_samples]
            + _colvec(cond_means[:, num_samples], _np=np)
        )
        # Samples [y_r] from [f_r]
        frmat = fsamples[1:]
        frm1mat = fsamples[:-1]
        arvec = _colvec(
            np.minimum(np.flip(avec[:t_obs]), -MIN_POSTERIOR_VARIANCE), _np=np
        )
        ysamples = ((frmat - frm1mat) / arvec - 1.0) * (1.0 / gamma) + frmat
    else:
        # Nothing to condition on
        fsamples = np.concatenate(tuple(reversed(f_rows)), axis=0)
        ysamples = np.concatenate(tuple(reversed(y_rows)), axis=0)
    return {"f": fsamples, "y": ysamples}


def issm_likelihood_slow_computations(
    targets: List[np.ndarray],
    issm_params: Dict,
    r_min: int,
    r_max: int,
    skip_c_d: bool = False,
    profiler: Optional[SimpleProfiler] = None,
) -> Dict:
    """
    Naive implementation of `issm_likelihood_computations`, which does not
    require precomputations, but is much slower. Here, results are computed
    one datapoint at a time, instead of en bulk.

    This code is used in unit testing, and called from `sample_posterior_joint`.
    """
    num_configs = len(targets)
    num_res = r_max + 1 - r_min
    assert num_configs > 0, "targets must not be empty"
    assert num_res > 0, f"r_min = {r_min} must be <= r_max = {r_max}"
    compute_wtw = targets[0].shape[1] == 1
    alphas = _flatvec(issm_params["alpha"])
    betas = _flatvec(issm_params["beta"])
    gamma = issm_params["gamma"]
    n = getval(alphas.shape[0])
    assert n == num_configs, f"alpha.size = {n} != {num_configs}"
    n = getval(betas.shape[0])
    assert n == num_configs, f"beta.size = {n} != {num_configs}"
    # Outer loop over configurations
    c_lst = []
    d_lst = []
    vtv_lst = []
    wtv_lst = []
    wtw_lst = []
    num_data = 0
    for i, ymat in enumerate(targets):
        alpha = alphas[i]
        alpha_m1 = alpha - 1.0
        beta = betas[i]
        ydim = ymat.shape[0]
        if profiler is not None:
            profiler.start("issm_part1")
        num_data += ydim
        r_obs = r_min + ydim  # Observed in range(r_min, r_obs)
        assert 0 < ydim <= num_res, f"len(y[{i}]) = {ydim}, num_res = {num_res}"
        if not skip_c_d:
            # c_i, d_i
            if ydim < num_res:
                lrvec = (
                    anp.array([np.log(r) for r in range(r_obs, r_max + 1)]) * alpha_m1
                    + beta
                )
                c_scal = alpha * anp.exp(logsumexp(lrvec))
                d_scal = anp.square(gamma * alpha) * anp.exp(logsumexp(lrvec * 2.0))
                c_lst.append(c_scal)
                d_lst.append(d_scal)
            else:
                c_lst.append(0.0)
                d_lst.append(0.0)
        # Inner loop for v_i, w_i
        if profiler is not None:
            profiler.stop("issm_part1")
            profiler.start("issm_part2")
        yprev = ymat[-1].reshape((1, -1))  # y_{j-1} (vector)
        vprev = 1.0  # v_{j-1} (scalar)
        wprev = yprev  # w_{j-1} (row vector)
        vtv = vprev * vprev  # scalar
        wtv = wprev * vprev  # row vector
        if compute_wtw:
            wtw = wprev * wprev  # shape (1, 1)
        for j in range(1, ydim):
            ycurr = ymat[ydim - j - 1].reshape((1, -1))  # y_j (row vector)
            # a_{j-1}
            ascal = alpha * anp.exp(np.log(r_obs - j) * alpha_m1 + beta)
            escal = gamma * ascal + 1.0
            vcurr = escal * vprev  # v_j
            wcurr = escal * wprev + ycurr - yprev + ascal  # w_j
            vtv = vtv + vcurr * vcurr
            wtv = wtv + wcurr * vcurr
            if compute_wtw:
                wtw = wtw + wcurr * wcurr
            yprev = ycurr
            vprev = vcurr
            wprev = wcurr
        vtv_lst.append(vtv)
        wtv_lst.append(wtv)
        if compute_wtw:
            assert wtw.shape == (1, 1)
            wtw_lst.append(wtw.item())
        if profiler is not None:
            profiler.stop("issm_part2")
    # Compile results
    result = {
        "num_data": num_data,
        "vtv": anp.array(vtv_lst),
        "wtv": anp.vstack(wtv_lst),
    }
    if compute_wtw:
        result["wtw"] = anp.array(wtw_lst)
    if not skip_c_d:
        result["c"] = anp.array(c_lst)
        result["d"] = anp.array(d_lst)
    return result


def _update_posterior_internal(
    poster_state: Dict, kernel, feature, d_new, s_new, r2_new
) -> Dict:
    assert "r2vec" in poster_state and "r4vec" in poster_state
    features = poster_state["features"]
    r2vec = poster_state["r2vec"]
    r4vec = poster_state["r4vec"]
    svec = poster_state["svec"]
    chol_fact = poster_state["chol_fact"]
    # New row of L: [evec * s_new, l_new]
    feature = _rowvec(feature, _np=np)
    kvec = _flatvec(kernel(features, feature), _np=np)
    evec = _flatvec(
        spl.solve_triangular(chol_fact, _colvec(kvec * svec), lower=True), _np=np
    )
    kscal = _flatvec(kernel.diagonal(feature), _np=np)[0]
    khat_min_esq = kscal + d_new - _squared_norm(evec, _np=np)
    l_new = np.sqrt(khat_min_esq * np.square(s_new) + 1.0)
    # New entry of r_4
    pref = s_new / l_new
    r4_new = pref * (
        _inner_product(kvec, r2vec, _np=np)
        + khat_min_esq * r2_new
        - _inner_product(evec, r4vec, _np=np)
    )
    # Update of p
    p_new = r2_new - pref * r4_new
    # L^{-T} e
    ltinv_evec = _flatvec(
        spl.solve_triangular(chol_fact, _colvec(evec, _np=np), lower=True, trans="T"),
        _np=np,
    )
    return {
        "evec": evec,
        "ltinv_evec": ltinv_evec,
        "l_new": l_new,
        "r4_new": r4_new,
        "p_new": p_new,
    }


def update_posterior_state(
    poster_state: Dict, kernel, feature, d_new, s_new, r2_new
) -> Dict:
    """
    Incremental update of posterior state, given data for one additional
    configuration. The new datapoint gives rise to a new row/column of the
    Cholesky factor. r2vec and svec are extended by `r2_new`, `s_new`
    respectively. r4vec and pvec are extended and all entries change. The new
    datapoint is represented by `feature`, `d_new`, `s_new`, `r2_new`.

    Note: The field `criterion` is not updated, but set to np.nan.

    :param poster_state: Posterior state for data
    :param kernel: Kernel function
    :param feature: Features for additional config
    :param d_new: See above
    :param s_new: See above
    :param r2_new: See above
    :return: Updated posterior state
    """
    features = poster_state["features"]
    assert "r2vec" in poster_state and "r4vec" in poster_state
    r2vec = poster_state["r2vec"]
    r4vec = poster_state["r4vec"]
    svec = poster_state["svec"]
    pvec = poster_state["pmat"]
    assert pvec.shape[1] == 1, "Cannot update fantasizing posterior_state"
    pvec = _flatvec(pvec, _np=np)
    chol_fact = poster_state["chol_fact"]
    feature = _rowvec(feature, _np=np)
    # Update computations:
    result = _update_posterior_internal(
        poster_state, kernel, feature, d_new, s_new, r2_new
    )
    # Put together new state variables
    # Note: Criterion not updated, but invalidated
    new_poster_state = {
        "criterion": np.nan,
        "features": np.concatenate((features, feature), axis=0),
        "svec": np.concatenate((svec, np.array([s_new]))),
        "r2vec": np.concatenate((r2vec, np.array([r2_new]))),
    }
    evec = result["evec"]
    ltinv_evec = result["ltinv_evec"]
    l_new = result["l_new"]
    r4_new = result["r4_new"]
    p_new = result["p_new"]
    new_poster_state["r4vec"] = np.concatenate(
        [r4vec + evec * r2_new, np.array([r4_new])]
    )
    new_poster_state["pmat"] = _colvec(
        np.concatenate([pvec - (ltinv_evec * svec) * p_new, np.array([p_new])]), _np=np
    )
    lvec = _rowvec(evec, _np=np) * s_new
    zerovec = _colvec(np.zeros_like(lvec), _np=np)
    lscal = np.array([l_new]).reshape((1, 1))
    new_poster_state["chol_fact"] = np.concatenate(
        (
            np.concatenate((chol_fact, lvec), axis=0),
            np.concatenate((zerovec, lscal), axis=0),
        ),
        axis=1,
    )
    return new_poster_state


def update_posterior_pvec(
    poster_state: Dict, kernel, feature, d_new, s_new, r2_new
) -> np.ndarray:
    """
    Part of `update_posterior_state`, just returns the new p vector.

    :param poster_state: See `update_posterior_state`
    :param kernel:  See `update_posterior_state`
    :param feature:  See `update_posterior_state`
    :param d_new:  See `update_posterior_state`
    :param s_new:  See `update_posterior_state`
    :param r2_new:  See `update_posterior_state`
    :return: New p vector, as flat vector

    """
    # Update computations:
    result = _update_posterior_internal(
        poster_state, kernel, feature, d_new, s_new, r2_new
    )
    svec = poster_state["svec"]
    pvec = poster_state["pmat"]
    assert pvec.shape[1] == 1, "Cannot update fantasizing posterior_state"
    pvec = _flatvec(pvec, _np=np)
    ltinv_evec = result["ltinv_evec"]
    p_new = result["p_new"]
    return np.concatenate((pvec - (ltinv_evec * svec) * p_new, np.array([p_new])))
