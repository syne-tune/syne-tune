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
from typing import List, Dict

import numpy as np
import autograd.numpy as anp
from autograd.scipy.linalg import solve_triangular
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.exponential_decay import (
    ExponentialDecayResourcesKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.custom_op import (
    cholesky_factorization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.issm import (
    _flatvec,
    _colvec,
    _rowvec,
    _squared_norm,
    _inner_product,
    predict_posterior_marginals,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_impl import (
    EPS,
)


class ZeroKernel(KernelFunction):
    """
    Constant zero kernel. This works only in the context used here, we do
    return matrices or vectors, but zero scalars.

    """

    def __init__(self, dimension: int, **kwargs):
        super().__init__(dimension, **kwargs)

    def forward(self, X1, X2, **kwargs):
        return 0.0

    def diagonal(self, X):
        return 0.0

    def diagonal_depends_on_X(self):
        return False

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass


class ZeroMean(MeanFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return 0.0

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass


class ExponentialDecayBaseKernelFunction(KernelFunction):
    """
    Implements exponential decay kernel k_r(r, r') from the Freeze-Thaw
    paper, corresponding to :class:`ExponentialDecayResourcesKernelFunction`
    with delta=0 and no x attributes.

    Note: Inputs r lie in [r_min, r_max]. Optionally, they are normalized to
    [0, 1].

    """

    def __init__(
        self, r_max: int, r_min: int = 1, normalize_inputs: bool = False, **kwargs
    ):
        super().__init__(dimension=1, **kwargs)
        self.kernel = ExponentialDecayResourcesKernelFunction(
            kernel_x=ZeroKernel(0), mean_x=ZeroMean(), delta_fixed_value=0.0
        )
        assert r_max > r_min
        self.r_min = r_min
        self.r_max = r_max
        self.lower = r_min - 0.5 + EPS
        self.width = r_max - r_min + 1 - 2 * EPS
        self.normalize_inputs = normalize_inputs

    def _normalize(self, X):
        return (X - self.lower) / self.width

    def forward(self, X1, X2):
        same_12 = X2 is X1
        if self.normalize_inputs:
            X1 = self._normalize(X1)
            if same_12:
                X2 = X1
            else:
                X2 = self._normalize(X2)
        return self.kernel(X1, X2)

    def diagonal(self, X):
        if self.normalize_inputs:
            X = self._normalize(X)
        return self.kernel.diagonal(X)

    def diagonal_depends_on_X(self):
        return self.kernel.diagonal_depends_on_X()

    def param_encoding_pairs(self):
        return self.kernel.param_encoding_pairs()

    def get_params(self):
        return self.kernel.get_params()

    def set_params(self, param_dict):
        self.kernel.set_params(param_dict)

    def mean_function(self, X):
        if self.normalize_inputs:
            X = self._normalize(X)
        return self.kernel.mean_function(X)


def logdet_cholfact_cov_resource(likelihood: Dict) -> float:
    """
    Computes the additional log(det(Lbar)) term. This is
    sum_i log(det(Lbar_i)), where Lbar_i is upper left submatrix of
    `likelihood['lfact_all']`, with size `likelihood['ydims'][i]`.

    :param likelihood: Result of `resource_kernel_likelihood_computations`
    :return: log(det(Lbar))
    """
    lfact_all = likelihood["lfact_all"]
    ydims = likelihood["ydims"]
    dim = max(ydims)
    log_diag = anp.log(anp.diag(lfact_all))
    # Weights:
    #   w_j = sum_i I[ydims[i] > j], j = 0, 1, ...
    weights = anp.sum(_colvec(anp.array(ydims)) > _rowvec(anp.arange(dim)), axis=0)
    return _inner_product(log_diag[:dim], weights)


def resource_kernel_likelihood_precomputations(targets: List[np.ndarray]) -> Dict:
    """
    Precomputations required by `resource_kernel_likelihood_computations`.

    Importantly, `prepare_data` orders datapoints by nonincreasing number of
    targets `ydims[i]`. For `0 <= j < ydim_max`, `ydim_max = ydims[0] =
    max(ydims)`, `num_configs[j]` is the number of datapoints i for which
    `ydims[i] > j`.
    `yflat` is a flat matrix (rows corresponding to fantasy samples; column
    vector if no fantasizing) consisting of `ydim_max` parts, where part j is of
    size `num_configs[j]` and contains `y[j]` for targets of those i counted in
    `num_configs[j]`.

    :param targets: Targets from data representation returned by
        `prepare_data`
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
    # Attention: When comparing this to `issm.issm_likelihood_precomputations`,
    # `targets` maps to `yflat` in the same ordering (the index is still r),
    # whereas there the index j of `deltay` runs in the opposite direction of
    # the index r of `targets`
    yflat_rows = []
    for pos, num in enumerate(num_configs):
        yflat_rows.extend([y[pos].reshape((1, -1)) for y in targets[:num]])
    yflat = np.vstack(yflat_rows)
    assert yflat.shape[0] == total_size
    return {"ydims": ydims, "num_configs": num_configs, "yflat": yflat}


# TODO: This code is complex. If it does not run faster than
# `resource_kernel_likelihood_slow_computations`, remove it.
def resource_kernel_likelihood_computations(
    precomputed: Dict,
    res_kernel: ExponentialDecayBaseKernelFunction,
    noise_variance,
    skip_c_d: bool = False,
) -> Dict:
    """
    Given `precomputed` from `resource_kernel_likelihood_precomputations` and
    resource kernel function `res_kernel`, compute quantities required for
    inference and marginal likelihood computation, pertaining to the likelihood
    of a additive model, as in the Freeze-Thaw paper.

    Note that `res_kernel` takes raw (unnormalized) r as inputs. The code here
    works for any resource kernel and mean function, not just for
    :class:`ExponentialDecayBaseKernelFunction`.

    Results returned are:
    - c: n vector [c_i]
    - d: n vector [d_i], positive
    - vtv: n vector [|v_i|^2]
    - wtv: (n, F) matrix[(W_i)^T v_i], F number of fantasy samples
    - wtw: n vector [|w_i|^2] (only if no fantasizing)
    - lfact_all: Cholesky factor for kernel matrix
    - ydims: Target vector sizes (copy from `precomputed`)

    :param precomputed: Output of `resource_kernel_likelihood_precomputations`
    :param res_kernel: Kernel k(r, r') over resources
    :param noise_variance: Noise variance sigma^2
    :param skip_c_d: If True, c and d are not computed
    :return: Quantities required for inference and learning criterion

    """
    num_configs = precomputed["num_configs"]
    num_all_configs = num_configs[0]
    r_min, r_max = res_kernel.r_min, res_kernel.r_max
    num_res = r_max + 1 - r_min
    assert num_all_configs > 0, "targets must not be empty"
    assert num_res > 0, f"r_min = {r_min} must be <= r_max = {r_max}"
    num_fantasy_samples = precomputed["yflat"].shape[1]
    compute_wtw = num_fantasy_samples == 1

    # Compute Cholesky factor for largest target vector size, or for full size
    ydims = precomputed["ydims"]
    rvals = _colvec(anp.arange(r_min, r_min + num_res))
    means_all = _flatvec(res_kernel.mean_function(rvals))
    amat = res_kernel(rvals, rvals) / noise_variance + anp.diag(anp.ones(num_res))
    # TODO: Do we need AddJitterOp here?
    lfact_all = cholesky_factorization(amat)  # L (Cholesky factor)

    # Loop over ydim
    yflat = precomputed["yflat"]
    off = num_all_configs
    ilscal = 1.0 / lfact_all[0, 0]
    vvec = anp.array([ilscal]).reshape((1, 1))
    # `yflat` is a (*, F) matrix, where F == `num_fantasy_samples`. These
    # matrices are flattened out as rows of `wmat`, and reshaped back before
    # writing into `wtv_lst`
    wmat = _rowvec(yflat[:off, :] - means_all[0]) * ilscal
    # Note: We need the detour via `wtv_lst`, etc, because `autograd` does not
    # support overwriting the content of an `ndarray`. Their role is to collect
    # parts of the final vectors, in reverse ordering
    wtv_lst = []
    wtw_lst = []
    num_prev = off
    for ydim, num in enumerate(num_configs[1:], start=1):
        if num < num_prev:
            # These parts are done:
            pos = num * num_fantasy_samples
            wdone = wmat[:, pos:]
            wtv_part = anp.reshape(anp.matmul(vvec, wdone), (-1, num_fantasy_samples))
            wtv_lst.append(wtv_part)
            if compute_wtw:
                wtw_lst.append(_flatvec(anp.sum(anp.square(wdone), axis=0)))
            wmat = wmat[:, :pos]
            num_prev = num
        # Update W matrix
        rhs = _rowvec(yflat[off : (off + num), :] - means_all[ydim])
        off += num
        lvec = _rowvec(lfact_all[ydim, :ydim])
        ilscal = 1.0 / lfact_all[ydim, ydim]
        w_new = (rhs - anp.matmul(lvec, wmat)) * ilscal
        wmat = anp.concatenate((wmat, w_new), axis=0)
        # Update v vector (row vector)
        v_new = anp.array([(1.0 - _inner_product(lvec, vvec)) * ilscal]).reshape((1, 1))
        vvec = anp.concatenate((vvec, v_new), axis=1)
    wtv_part = anp.reshape(anp.matmul(vvec, wmat), (-1, num_fantasy_samples))
    wtv_lst.append(wtv_part)
    wtv_all = anp.concatenate(tuple(reversed(wtv_lst)), axis=0)
    if compute_wtw:
        wtw_lst.append(_flatvec(anp.sum(anp.square(wmat), axis=0)))
        wtw_all = anp.concatenate(tuple(reversed(wtw_lst)), axis=0)
    vtv_for_ydim = anp.cumsum(anp.square(vvec))
    vtv_all = anp.array([vtv_for_ydim[ydim - 1] for ydim in ydims])
    # Compile results
    result = {
        "num_data": sum(ydims),
        "vtv": vtv_all,
        "wtv": wtv_all,
        "lfact_all": lfact_all,
        "means_all": means_all,
        "ydims": ydims,
    }
    if compute_wtw:
        result["wtw"] = wtw_all
    if not skip_c_d:
        result["c"] = anp.zeros(num_all_configs)
        result["d"] = anp.zeros(num_all_configs)
    return result


# TODO: It is not clear whether this code is slower, and it is certainly
# simpler.
def resource_kernel_likelihood_slow_computations(
    targets: List[np.ndarray],
    res_kernel: ExponentialDecayBaseKernelFunction,
    noise_variance,
    skip_c_d: bool = False,
) -> Dict:
    """
    Naive implementation of `resource_kernel_likelihood_computations`, which
    does not require precomputations, but is somewhat slower. Here, results are
    computed one datapoint at a time, instead of en bulk.

    This code is used in unit testing only.
    """
    num_configs = len(targets)
    r_min, r_max = res_kernel.r_min, res_kernel.r_max
    num_res = r_max + 1 - r_min
    assert num_configs > 0, "targets must not be empty"
    compute_wtw = targets[0].shape[1] == 1
    # Compute Cholesky factor for largest target vector size
    ydims = [y.shape[0] for y in targets]
    rvals = _colvec(anp.arange(r_min, r_min + num_res))
    means_all = _flatvec(res_kernel.mean_function(rvals))
    amat = res_kernel(rvals, rvals) / noise_variance + anp.diag(anp.ones(num_res))
    # TODO: Do we need AddJitterOp here?
    lfact_all = cholesky_factorization(amat)  # L (Cholesky factor)
    # Outer loop over configurations
    vtv_lst = []
    wtv_lst = []
    wtw_lst = []
    num_data = 0
    for i, (ymat, ydim) in enumerate(zip(targets, ydims)):
        assert 0 < ydim <= num_res, f"len(y[{i}]) = {ydim}, num_res = {num_res}"
        num_data += ydim
        lfact = lfact_all[:ydim, :ydim]
        rhs = anp.ones((ydim, 1))
        vvec = _flatvec(solve_triangular(lfact, rhs, lower=True))
        means = means_all[:ydim]
        rhs = ymat - _colvec(means)
        wmat = solve_triangular(lfact, rhs, lower=True)
        vtv_lst.append(_squared_norm(vvec))
        wtv_lst.append(anp.matmul(_rowvec(vvec), wmat))
        if compute_wtw:
            wtw_lst.append(_squared_norm(wmat))
    # Compile results
    result = {
        "num_data": num_data,
        "vtv": anp.array(vtv_lst),
        "wtv": anp.vstack(wtv_lst),
        "lfact_all": lfact_all,
        "means_all": means_all,
        "ydims": ydims,
    }
    if compute_wtw:
        result["wtw"] = anp.array(wtw_lst)
    if not skip_c_d:
        result["c"] = anp.zeros(num_configs)
        result["d"] = anp.zeros(num_configs)
    return result


def predict_posterior_marginals_extended(
    poster_state: Dict,
    mean,
    kernel,
    test_features,
    resources: List[int],
    res_kernel: ExponentialDecayBaseKernelFunction,
):
    """
    These are posterior marginals on f_r = h + g_r variables, where
    (x, r) are zipped from `test_features`, `resources`.
    `posterior_means` is a (n, F) matrix, where F is the number of fantasy
    samples, or F == 1 without fantasizing.

    :param poster_state: Posterior state
    :param mean: Mean function
    :param kernel: Kernel function
    :param test_features: Feature matrix for test points (not extended)
    :param resources: Resource values corresponding to rows of
        `test_features`
    :param res_kernel: Kernel k(r, r') over resources
    :return: posterior_means, posterior_variances

    """
    num_test = test_features.shape[0]
    assert len(resources) == num_test, (
        f"test_features.shape[0] = {num_test} != {len(resources)} " + "= len(resources)"
    )
    # Predictive marginals over h
    h_means, h_variances = predict_posterior_marginals(
        poster_state, mean, kernel, test_features
    )
    # Convert into predictive marginals over f_r
    rvals = _colvec(anp.array(resources))
    g_means = _colvec(res_kernel.mean_function(rvals))
    g_variances = _flatvec(res_kernel.diagonal(rvals))
    posterior_means = h_means + g_means
    posterior_variances = h_variances + g_variances
    return posterior_means, posterior_variances


def sample_posterior_joint(
    poster_state: Dict,
    mean,
    kernel,
    feature,
    targets: np.ndarray,
    res_kernel: ExponentialDecayBaseKernelFunction,
    noise_variance,
    lfact_all,
    means_all,
    random_state: RandomState,
    num_samples: int = 1,
) -> Dict:
    """
    Given `poster_state` for some data plus one additional configuration
    with data (`feature`, `targets`), draw joint samples of unobserved
    targets for this configuration. `targets` may be empty, but must not
    be complete (there must be some unobserved targets). The additional
    configuration must not be in the dataset used to compute `poster_state`.

    If `targets` correspond to resource values range(r_min, r_obs), we
    sample latent target values y_r corresponding to range(r_obs, r_max+1),
    returning a dict with [y_r] under `y` (matrix with `num_samples`
    columns).

    :param poster_state: Posterior state for data
    :param mean: Mean function
    :param kernel: Kernel function
    :param feature: Features for additional config
    :param targets: Target values for additional config
    :param res_kernel: Kernel k(r, r') over resources
    :param noise_variance: Noise variance sigma^2
    :param lfact_all: Cholesky factor of complete resource kernel matrix
    :param means_all: See `lfact_all`
    :param random_state: numpy.random.RandomState
    :param num_samples: Number of joint samples to draw (default: 1)
    :return: See above
    """
    r_min, r_max = res_kernel.r_min, res_kernel.r_max
    num_res = r_max + 1 - r_min
    targets = _colvec(targets, _np=np)
    ydim = targets.size
    assert ydim < num_res, f"len(targets) = {ydim} must be < {num_res}"
    assert lfact_all.shape == (
        num_res,
        num_res,
    ), f"lfact_all.shape = {lfact_all.shape}, must be {(num_res, num_res)}"
    assert (
        means_all.size == num_res
    ), f"means_all.size = {means_all.size}, must be {num_res}"

    # Posterior mean and variance of h for additional config
    post_mean, post_variance = predict_posterior_marginals(
        poster_state, mean, kernel, _rowvec(feature)
    )
    post_mean = post_mean[0]
    post_variance = post_variance[0]
    # Draw samples from joint distribution
    epsmat = random_state.normal(size=(num_res, num_samples))
    joint_samples = anp.matmul(lfact_all, epsmat) * anp.sqrt(
        noise_variance
    ) + anp.reshape(means_all, (-1, 1))
    hvec = (
        random_state.normal(size=(1, num_samples)) * anp.sqrt(post_variance) + post_mean
    )
    joint_samples = joint_samples + hvec
    if ydim > 0:
        # There are observed targets, so have to transform the joint sample
        # into a conditional one
        targets_samp = joint_samples[:ydim, :]
        lfact = lfact_all[:ydim, :ydim]  # L_Q
        rhs = anp.ones((ydim, 1))
        vvec = solve_triangular(lfact, rhs, lower=True)  # v
        vtv = _squared_norm(vvec)  # alpha
        # w_hat - w
        w_delta = solve_triangular(lfact, targets - targets_samp, lower=True)
        kappa = post_variance / noise_variance
        fact = kappa / (vtv * kappa + 1.0)
        # rho_hat - rho
        rho_delta = anp.matmul(_rowvec(vvec), w_delta) * fact
        tmpmat = w_delta - vvec * rho_delta
        lfact_pq = lfact_all[ydim:, :ydim]  # L_{P, Q}
        ysamples = joint_samples[ydim:, :] + anp.matmul(lfact_pq, tmpmat) + rho_delta
    else:
        # Nothing to condition on
        ysamples = joint_samples
    return {"y": ysamples}
