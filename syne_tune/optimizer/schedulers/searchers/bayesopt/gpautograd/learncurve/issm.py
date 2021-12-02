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
from typing import List, Dict, Optional

import numpy as np
import autograd.numpy as anp
from autograd.scipy.special import logsumexp
from autograd.scipy.linalg import solve_triangular
from autograd.tracer import getval
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import NUMERICAL_JITTER, MIN_POSTERIOR_VARIANCE
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.custom_op \
    import cholesky_factorization
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler


def prepare_data(
        state: TuningJobState, configspace_ext: ExtendedConfiguration,
        active_metric: str, normalize_targets: bool = False) -> Dict:
    # Group w.r.t. configs
    # data maps config_to_tuple(config) -> (config, observed), where
    # observed = [(r_j, y_j)], r_j resource values, y_j reward values. The
    # key is easily hashable, while config may not be.
    # The lists returned ('configs', 'targets') are sorted in decreasing
    # order w.r.t. the number of targets.
    r_min, r_max = configspace_ext.resource_attr_range
    hp_ranges = configspace_ext.hp_ranges
    data_lst = []
    targets = []

    for candidate in state.candidate_evaluations:
        metric_vals = candidate.metrics[active_metric]
        assert isinstance(metric_vals, dict)
        observed = list(sorted(
            ((int(k), v) for k, v in metric_vals.items()),
            key=lambda x: x[0]))
        data_lst.append((candidate.candidate, observed))
        targets += [x[1] for x in observed]
    mean = 0.0
    std = 1.0
    if normalize_targets:
        std = max(np.std(targets), 1e-9)
        mean = np.mean(targets)
    configs = [x[0] for x in data_lst]
    targets = []
    for config, observed in data_lst:
        # Observations must be from r_min without any missing
        obs_res = [x[0] for x in observed]
        num_obs = len(observed)
        test = list(range(r_min, r_min + num_obs))
        assert obs_res == test, \
            f"Config {config} has observations at {obs_res}, but " +\
            f"we need observations at {test}"
        targets.append([(x[1] - mean) / std for x in observed])
    # Sort in decreasing order w.r.t. number of targets
    configs, targets = zip(*sorted(
        zip(configs, targets), key=lambda x: -len(x[1])))
    features = hp_ranges.to_ndarray_matrix(configs)
    result = {
        'configs': configs,
        'features': features,
        'targets': targets,
        'r_min': r_min,
        'r_max': r_max}
    if normalize_targets:
        result['mean_targets'] = mean
        result['std_targets'] = std
    return result

def issm_likelihood_precomputations(targets: List[tuple], r_min: int) -> Dict:
    """
    Precomputations required by `issm_likelihood_computations`.

    Importantly, `prepare_data` orders datapoints by nonincreasing number of
    targets `ydims[i]`. For `0 <= j < ydim_max`, `ydim_max = ydims[0] =
    max(ydims)`, `num_configs[j]` is the number of datapoints i for which
    `ydims[i] > j`.
    `deltay` is a flat vector consisting of `ydim_max` parts, where part j is
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
    ydims = [len(y) for y in targets]
    ydim_max = ydims[0]
    num_configs = list(np.sum(
        np.array(ydims).reshape((-1, 1)) > np.arange(ydim_max).reshape((1, -1)),
        axis=0).reshape((-1,)))
    assert num_configs[0] == len(targets), (num_configs, len(targets))
    assert num_configs[-1] > 0, num_configs
    total_size = sum(num_configs)
    assert total_size == sum(ydims)
    deltay = np.zeros(total_size)
    yprev = np.array([y[-1] for y in targets])
    off = num_configs[0]
    deltay[:off] = yprev
    log_r = []
    for pos, num in enumerate(num_configs[1:], start=1):
        ycurr = np.array([y[-(pos + 1)] for y in targets[:num]])
        deltay[off:(off + num)] = ycurr - yprev[:num]
        off += num
        yprev = ycurr
        logr_curr = [np.log(ydim + r_min - pos) for ydim in ydims[:num]]
        log_r.extend(logr_curr)
    assert off == total_size
    assert len(log_r) == total_size - num_configs[0]
    return {
        'ydims': ydims,
        'num_configs': num_configs,
        'deltay': deltay,
        'logr': np.array(log_r)}


def issm_likelihood_computations(
        precomputed: Dict, issm_params: Dict, r_min: int, r_max: int,
        skip_c_d: bool = False,
        profiler: Optional[SimpleProfiler] = None) -> Dict:
    """
    Given `precomputed` from `issm_likelihood_precomputations` and ISSM
    parameters `issm_params`, compute quantities required for inference and
    marginal likelihood computation, pertaining to the ISSM likelihood.

    The index for r is range(r_min, r_max + 1). Observations must be contiguous
    from r_min. The ISSM parameters are:
    - alpha: n-vector, negative
    - beta: n-vector
    - gamma: scalar, positive

    Likelihood computations are detailed in the report

    SyneDocs/MatthiasS/mlp_tuning/gpmodel_hb.tex, Section 5.3.

    Results returned are:
    - c: n-vector [c_i], negative
    - d: n-vector [d_i], positive
    - vtv: n-vector [|v_i|^2]
    - wtv: n-vector [(w_i)^T v_i]
    - wtw: n-vector [|w_i|^2]

    :param precomputed: Output of `issm_likelihood_precomputations`
    :param issm_params: Parameters of ISSM likelihood
    :param r_min: Smallest resource value
    :param r_max: Largest resource value
    :param skip_c_d: If True, c and d are not computed
    :return: Quantities required for inference and learning criterion

    """
    num_all_configs = precomputed['num_configs'][0]
    num_res = r_max + 1 - r_min
    assert num_all_configs > 0, "targets must not be empty"
    assert num_res > 0, f"r_min = {r_min} must be <= r_max = {r_max}"
    alphas = anp.reshape(issm_params['alpha'], (-1,))
    betas = anp.reshape(issm_params['beta'], (-1,))
    gamma = issm_params['gamma']
    n = getval(alphas.size)
    assert n == num_all_configs, f"alpha.size = {n} != {num_all_configs}"
    n = getval(betas.size)
    assert n == num_all_configs, f"beta.size = {n} != {num_all_configs}"

    if not skip_c_d:
        # We could probably refactor this to fit into the loop below, but it
        # seems subdominant
        if profiler is not None:
            profiler.start('issm_part1')
        c_lst = []
        d_lst = []
        for i, ydim in enumerate(precomputed['ydims']):
            alpha = alphas[i]
            alpha_m1 = alpha - 1.0
            beta = betas[i]
            r_obs = r_min + ydim  # Observed in range(r_min, r_obs)
            assert 0 < ydim <= num_res,\
                f"len(y[{i}]) = {ydim}, num_res = {num_res}"
            # c_i, d_i
            if ydim < num_res:
                lrvec = anp.array(
                    [np.log(r) for r in range(r_obs, r_max + 1)]) *\
                        alpha_m1 + beta
                c_scal = alpha * anp.exp(logsumexp(lrvec))
                d_scal = anp.square(gamma * alpha) * anp.exp(
                    logsumexp(lrvec * 2.0))
                c_lst.append(c_scal)
                d_lst.append(d_scal)
            else:
                c_lst.append(0.0)
                d_lst.append(0.0)
        if profiler is not None:
            profiler.stop('issm_part1')

    # Loop over ydim
    if profiler is not None:
        profiler.start('issm_part2')
    deltay = precomputed['deltay']
    logr = precomputed['logr']
    off_dely = num_all_configs
    vvec = anp.ones(off_dely)
    wvec = deltay[:off_dely]  # [y_0]
    vtv = anp.ones(off_dely)
    wtv = wvec.copy()
    wtw = anp.square(wvec)
    # Note: We need the detour via `vtv_lst`, etc, because `autograd` does not
    # support overwriting the content of an `ndarray`. Their role is to collect
    # parts of the final vectors, in reverse ordering
    vtv_lst = []
    wtv_lst = []
    wtw_lst = []
    alpham1s = alphas - 1
    num_prev = off_dely
    for num in precomputed['num_configs'][1:]:
        if num < num_prev:
            # Size of working vectors is shrinking
            assert vtv.size == num_prev
            # These parts are done: Collect them in the lists
            vtv_lst.append(vtv[num:])
            wtv_lst.append(wtv[num:])
            wtw_lst.append(wtw[num:])
            # All vectors are resized to `num`, dropping the tails
            vtv = vtv[:num]
            wtv = wtv[:num]
            wtw = wtw[:num]
            alphas = alphas[:num]
            alpham1s = alpham1s[:num]
            betas = betas[:num]
            vvec = vvec[:num]
            wvec = wvec[:num]
            num_prev = num
        # [a_{j-1}]
        off_logr = off_dely - num_all_configs
        logr_curr = logr[off_logr:(off_logr + num)]
        avec = alphas * anp.exp(logr_curr * alpham1s + betas)
        evec = avec * gamma + 1  # [e_j]
        vvec = vvec * evec  # [v_j]
        deltay_curr = deltay[off_dely:(off_dely + num)]
        off_dely += num
        wvec = wvec * evec + deltay_curr + avec  # [w_j]
        vtv = vtv + anp.square(vvec)
        wtw = wtw + anp.square(wvec)
        wtv = wtv + wvec * vvec
    vtv_lst.append(vtv)
    wtv_lst.append(wtv)
    wtw_lst.append(wtw)
    vtv_all = anp.concatenate(tuple(reversed(vtv_lst)), axis=0)
    wtv_all = anp.concatenate(tuple(reversed(wtv_lst)), axis=0)
    wtw_all = anp.concatenate(tuple(reversed(wtw_lst)), axis=0)
    if profiler is not None:
        profiler.stop('issm_part2')

    # Compile results
    result = {
        'num_data': sum(precomputed['ydims']),
        'vtv': vtv_all,
        'wtv': wtv_all,
        'wtw': wtw_all}
    if not skip_c_d:
        result['c'] = anp.array(c_lst)
        result['d'] = anp.array(d_lst)
    return result


def _squared_norm(a):
    return anp.sum(anp.square(a))


def _inner_product(a, b):
    return anp.sum(anp.multiply(a, b))


def _colvec(a):
    return anp.reshape(a, (-1, 1))


def _rowvec(a):
    return anp.reshape(a, (1, -1))


def _flatvec(a):
    return anp.reshape(a, (-1,))


def posterior_computations(
        features, mean, kernel, issm_likelihood: Dict, noise_variance) -> Dict:
    """
    Computes posterior state (required for predictions) and negative log
    marginal likelihood.

    :param features: Input matrix X
    :param mean: Mean function
    :param kernel: Kernel function
    :param issm_likelihood: Outcome of `issm_likelihood_computations`
    :param noise_variance: Variance of ISSM innovations
    :return: Internal posterior state

    """
    num_data = issm_likelihood['num_data']
    kernel_mat = kernel(features, features)  # K
    dvec = issm_likelihood['d']
    s2vec = issm_likelihood['vtv'] / noise_variance  # S^2
    svec = _colvec(anp.sqrt(s2vec + NUMERICAL_JITTER))
    # A = I + S K_hat S = (I + S^2 D) + S K S
    dgvec = s2vec * dvec + 1.0
    amat = anp.multiply(svec, anp.multiply(
        kernel_mat, _rowvec(svec))) + anp.diag(dgvec)
    # TODO: Do we need AddJitterOp here?
    lfact = cholesky_factorization(amat)  # L (Cholesky factor)
    # r vectors
    muhat = _flatvec(mean(features)) - issm_likelihood['c']
    s2muhat = s2vec * muhat  # S^2 mu_hat
    r2vec = _colvec(issm_likelihood['wtv'] / noise_variance - s2muhat)
    r3vec = anp.matmul(kernel_mat, r2vec) + _colvec(dvec) * r2vec
    r4vec = solve_triangular(lfact, r3vec * svec, lower=True)
    # Prediction vector p
    svec = _flatvec(svec)
    r2vec = _flatvec(r2vec)
    pvec = r2vec - svec * _flatvec(
        solve_triangular(lfact, r4vec, lower=True, trans='T'))
    # Negative log marginal likelihood
    # Part sigma^{-2} |r_1|^2 - (r_2)^T r_3 + |r_4|^2
    r3vec = _flatvec(r3vec)
    r4vec = _flatvec(r4vec)
    # We use: r_1 = w - V mu_hat, sigma^{-2} V^T V = S^2, and
    # r_2 = sigma^{-2} V^T w - S^2 mu_hat, so that:
    # sigma^{-2} |r_1|^2
    # = sigma^{-2} |w|^2 + |S mu_hat|^2 - 2 sigma^{-2} w^T V mu_hat
    # = sigma^{-2} |w|^2 - |S mu_hat|^2 - 2 * (r_2)^T mu_hat
    # = sigma^{-2} |w|^2 - mu_hat^T (S^2 mu_hat + 2 r_2)
    part2 = 0.5 * (anp.sum(issm_likelihood['wtw']) / noise_variance -
                   _inner_product(muhat, s2muhat + 2.0 * r2vec) -
                   _inner_product(r2vec, r3vec) + _squared_norm(r4vec))
    part1 = anp.sum(anp.log(anp.abs(anp.diag(lfact)))) + \
            0.5 * num_data * anp.log(2 * anp.pi * noise_variance)
    criterion = part1 + part2
    return {
        'features': features,
        'chol_fact': lfact,
        'svec': svec,
        'pvec': pvec,
        'criterion': criterion,
        'likelihood': issm_likelihood}


def predict_posterior_marginals(
        poster_state: Dict, mean, kernel, test_features):
    k_tr_te = kernel(poster_state['features'], test_features)
    posterior_means = _flatvec(anp.matmul(
        _rowvec(poster_state['pvec']), k_tr_te)) + _flatvec(mean(test_features))
    qmat = solve_triangular(
        poster_state['chol_fact'], anp.multiply(
            _colvec(poster_state['svec']), k_tr_te), lower=True)
    posterior_variances = kernel.diagonal(test_features) - anp.sum(
        anp.square(qmat), axis=0)
    return posterior_means, _flatvec(anp.maximum(
        posterior_variances, MIN_POSTERIOR_VARIANCE))


def sample_posterior_marginals(
        poster_state: Dict, mean, kernel, test_features,
        random_state: RandomState, num_samples: int = 1):
    post_means, post_vars = predict_posterior_marginals(
        poster_state, mean, kernel, test_features)
    n01_mat = random_state.normal(
        size=(getval(post_means.shape[0]), num_samples))
    post_stds = _colvec(anp.sqrt(post_vars))
    return anp.multiply(post_stds, n01_mat) + _colvec(post_means)


# TODO!! Use precomputation and `issm_likelihood_computations` !!
def sample_posterior_joint(
        poster_state: Dict, mean, kernel, feature, targets: List,
        issm_params: Dict, r_min: int, r_max: int, random_state: RandomState,
        num_samples: int = 1) -> Dict:
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
    ydim = len(targets)
    t_obs = num_res - ydim
    assert t_obs > 0, f"len(targets) = {ydim} must be < {num_res}"
    # ISSM parameters
    alpha = issm_params['alpha'][0]
    alpha_m1 = alpha - 1.0
    beta = issm_params['beta'][0]
    gamma = issm_params['gamma']
    # Posterior mean and variance of h for additional config
    post_mean, post_variance = predict_posterior_marginals(
        poster_state, mean, kernel, _rowvec(feature))
    post_mean = post_mean[0]
    post_variance = post_variance[0]
    # Compute [a_t], [gamma^2 a_t^2]
    lrvec = anp.array(
        [np.log(r_max - t) for t in range(num_res - 1)]) * alpha_m1 + beta
    avec = alpha * anp.exp(lrvec)
    a2vec = anp.square(alpha * gamma) * anp.exp(lrvec * 2.0)
    # Draw the [eps_t] for all samples
    epsmat = random_state.normal(size=(num_res, num_samples))
    # Compute samples [f_t], [y_t], not conditioned on targets
    hvec = random_state.normal(
        size=(1, num_samples)) * anp.sqrt(post_variance) + post_mean
    f_rows = []
    y_rows = []
    fcurr = hvec
    for t in range(num_res - 1):
        eps_row = _rowvec(epsmat[t])
        f_rows.append(fcurr)
        y_rows.append(fcurr + eps_row)
        fcurr = fcurr - avec[t] * (eps_row * gamma + 1.0)
    eps_row = _rowvec(epsmat[-1])
    f_rows.append(fcurr)
    y_rows.append(fcurr + eps_row)
    if ydim > 0:
        # Condition on targets
        # Prior samples (reverse order t -> r)
        fsamples = anp.concatenate(reversed(f_rows[:(t_obs + 1)]), axis=0)
        # Compute c1 and d1 vectors (same for all samples)
        zeroscal = anp.zeros((1,))
        c1vec = anp.flip(anp.concatenate(
            (zeroscal, anp.cumsum(avec[:t_obs])), axis=None))
        d1vec = anp.flip(anp.concatenate(
            (zeroscal, anp.cumsum(a2vec[:t_obs])), axis=None))
        # Assemble targets for conditional means
        ymat = anp.concatenate(reversed(y_rows[t_obs:]), axis=0)
        ycols = anp.split(ymat, num_samples, axis=1)
        assert ycols[0].size == ydim  # Sanity check
        # v^T v, w^T v for sampled targets
        onevec = anp.ones((num_samples,))
        _issm_params = {
            'alpha': alpha * onevec,
            'beta': beta * onevec,
            'gamma': gamma}
        issm_likelihood = issm_likelihood_slow_computations(
            [_flatvec(v) for v in ycols], _issm_params, r_min, r_max,
            skip_c_d=True)
        vtv = issm_likelihood['vtv']
        wtv = issm_likelihood['wtv']
        # v^T v, w^T v for observed (last entry)
        issm_likelihood = issm_likelihood_slow_computations(
            [targets], issm_params, r_min, r_max)
        vtv = _rowvec(
            anp.concatenate((vtv, issm_likelihood['vtv']), axis=None))
        wtv = _rowvec(
            anp.concatenate((wtv, issm_likelihood['wtv']), axis=None))
        cscal = issm_likelihood['c'][0]
        dscal = issm_likelihood['d'][0]
        c1vec = _colvec(c1vec)
        d1vec = _colvec(d1vec)
        c2vec = cscal - c1vec
        d2vec = dscal - d1vec
        # Compute num_samples + 1 conditional mean vectors in one go
        denom = vtv * (post_variance + dscal) + 1.0
        cond_means = ((post_mean - c1vec) * (d2vec * vtv + 1.0) +
                      (d1vec + post_variance) * (c2vec * vtv + wtv)) / denom
        fsamples = fsamples - cond_means[:, :num_samples] + _colvec(
            cond_means[:, num_samples])
        # Samples [y_r] from [f_r]
        frmat = fsamples[1:]
        frm1mat = fsamples[:-1]
        arvec = _colvec(anp.minimum(
            anp.flip(avec[:t_obs]), -MIN_POSTERIOR_VARIANCE))
        ysamples = ((frmat - frm1mat) / arvec - 1.0) * (1.0 / gamma) + frmat
    else:
        # Nothing to condition on
        fsamples = anp.concatenate(reversed(f_rows), axis=0)
        ysamples = anp.concatenate(reversed(y_rows), axis=0)
    return {
        'f': fsamples,
        'y': ysamples}


def issm_likelihood_slow_computations(
        targets: List[tuple], issm_params: Dict, r_min: int, r_max: int,
        skip_c_d: bool = False,
        profiler: Optional[SimpleProfiler] = None) -> Dict:
    """
    Naive implementation of `issm_likelihood_computations`, which does not
    require precomputations, but is much slower. Here, results are computed
    one datapoint at a time, instead of en bulk.

    This code is used in unit testing only.
    """
    num_configs = len(targets)
    num_res = r_max + 1 - r_min
    assert num_configs > 0, "targets must not be empty"
    assert num_res > 0, f"r_min = {r_min} must be <= r_max = {r_max}"
    alphas = anp.reshape(issm_params['alpha'], (-1,))
    betas = anp.reshape(issm_params['beta'], (-1,))
    gamma = issm_params['gamma']
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
    for i, yvec in enumerate(targets):
        alpha = alphas[i]
        alpha_m1 = alpha - 1.0
        beta = betas[i]
        ydim = len(yvec)
        if profiler is not None:
            profiler.start('issm_part1')
        num_data += ydim
        r_obs = r_min + ydim  # Observed in range(r_min, r_obs)
        assert 0 < ydim <= num_res,\
            f"len(y[{i}]) = {ydim}, num_res = {num_res}"
        if not skip_c_d:
            # c_i, d_i
            if ydim < num_res:
                lrvec = anp.array(
                    [np.log(r) for r in range(r_obs, r_max + 1)]) *\
                        alpha_m1 + beta
                c_scal = alpha * anp.exp(logsumexp(lrvec))
                d_scal = anp.square(gamma * alpha) * anp.exp(
                    logsumexp(lrvec * 2.0))
                c_lst.append(c_scal)
                d_lst.append(d_scal)
            else:
                c_lst.append(0.0)
                d_lst.append(0.0)
        # Inner loop for v_i, w_i
        if profiler is not None:
            profiler.stop('issm_part1')
            profiler.start('issm_part2')
        yprev = yvec[-1]  # y_{j-1}
        vprev = 1.0  # v_{j-1}
        wprev = yprev  # w_{j-1}
        vtv = vprev * vprev
        wtv = wprev * vprev
        wtw = wprev * wprev
        for j in range(1, ydim):
            ycurr = yvec[ydim - j - 1]  # y_j
            # a_{j-1}
            ascal = alpha * anp.exp(np.log(r_obs - j) * alpha_m1 + beta)
            escal = gamma * ascal + 1.0
            vcurr = escal * vprev  # v_j
            wcurr = escal * wprev + ycurr - yprev + ascal  # w_j
            vtv = vtv + vcurr * vcurr
            wtv = wtv + wcurr * vcurr
            wtw = wtw + wcurr * wcurr
            yprev = ycurr
            vprev = vcurr
            wprev = wcurr
        vtv_lst.append(vtv)
        wtv_lst.append(wtv)
        wtw_lst.append(wtw)
        if profiler is not None:
            profiler.stop('issm_part2')
    # Compile results
    result = {
        'num_data': num_data,
        'vtv': anp.array(vtv_lst),
        'wtv': anp.array(wtv_lst),
        'wtw': anp.array(wtw_lst)}
    if not skip_c_d:
        result['c'] = anp.array(c_lst)
        result['d'] = anp.array(d_lst)
    return result
