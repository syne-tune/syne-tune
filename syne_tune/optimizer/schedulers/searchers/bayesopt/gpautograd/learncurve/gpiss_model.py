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
import logging
import numpy as np
import autograd.numpy as anp
from typing import Optional, List, Dict

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.likelihood \
    import MarginalLikelihood, LCModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.posterior_state \
    import IncrementalUpdateGPAdditivePosteriorState
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ScalarMeanFunction, MeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.optimization_utils \
    import apply_lbfgs_with_multiple_starts, create_lbfgs_arguments
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler

logger = logging.getLogger(__name__)


class GaussianProcessLearningCurveModel(object):
    """
    Represents joint Gaussian model of learning curves over a number of
    configurations. The model has an additive form:

        f(x, r) = g(r | x) + h(x),

    where h(x) is a Gaussian process model for function values at r_max, and
    the g(r | x) are independent Gaussian models. Right now, g(r | x) can be:

    - Innovation state space model (ISSM) of a particular power-law decay
        form. For this one, g(r_max | x) = 0 for all x. Used if
        `res_model` is of type :class:`ISSModelParameters`
    - Gaussian process model with exponential decay covariance function. This
        is essentially the model from the Freeze Thaw paper, see also
        :class:`ExponentialDecayResourcesKernelFunction`. Used if
        `res_model` is of type :class:`ExponentialDecayBaseKernelFunction`

    Importantly, inference scales cubically only in the number of
    configurations, not in the number of observations.

    Details about ISSMs in general are found in

        Hyndman, R. and Koehler, A. and Ord, J. and Snyder, R.
        Forecasting with Exponential Smoothing: The State Space Approach
        Springer, 2008

    :param kernel: Kernel function k(X, X')
    :param res_model: Model for g(r | x)
    :param mean: Mean function mu(X)
    :param initial_noise_variance: A scalar to initialize the value of the
        residual noise variance
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values

    """
    def __init__(
            self, kernel: KernelFunction, res_model: LCModel,
            mean: MeanFunction = None, initial_noise_variance: float = None,
            optimization_config: OptimizationConfig = None, random_seed=None,
            fit_reset_params: bool = True,
            use_precomputations: bool = True):

        if mean is None:
            mean = ScalarMeanFunction()
        if optimization_config is None:
            optimization_config = DEFAULT_OPTIMIZATION_CONFIG
        if random_seed is None:
            random_seed = 31415927
        self._random_state = np.random.RandomState(random_seed)
        self._states = None
        self.fit_reset_params = fit_reset_params
        self.optimization_config = optimization_config
        self._use_precomputations = use_precomputations
        self.likelihood = MarginalLikelihood(
            kernel=kernel,
            res_model=res_model,
            mean=mean,
            initial_noise_variance=initial_noise_variance)
        self.reset_params()

    @property
    def states(self) -> Optional[List[IncrementalUpdateGPAdditivePosteriorState]]:
        return self._states

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random_state

    @staticmethod
    def _debug_log_histogram(data: Dict):
        from collections import Counter

        histogram = Counter(len(y) for y in data['targets'])
        sorted_hist = sorted(histogram.items(), key=lambda x: x[0])
        logger.info(f"Histogram target size: {sorted_hist}")

    def fit(self, data: Dict, profiler: Optional[SimpleProfiler] = None):
        """
        Fit the (hyper)parameters of the model by optimizing the marginal
        likelihood, and set posterior states. `data` is obtained from
        `TuningJobState` by `issm.prepare_data`.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        :param data: Input points (features, configs), targets. May also have
            to contain precomputed values
        """
        assert not data['do_fantasizing'], \
            "data must not be for fantasizing. Call prepare_data with " +\
            "do_fantasizing=False"
        self._data_precomputations(data)
        if self.fit_reset_params:
            self.reset_params()
        if profiler is not None:
            self.likelihood.set_profiler(profiler)
            self._debug_log_histogram(data)
        n_starts = self.optimization_config.n_starts
        ret_infos = apply_lbfgs_with_multiple_starts(
            *create_lbfgs_arguments(
                criterion=self.likelihood, crit_args=[data],
                verbose=self.optimization_config.verbose),
            bounds=self.likelihood.box_constraints_internal(),
            random_state=self._random_state,
            n_starts=n_starts,
            tol=self.optimization_config.lbfgs_tol,
            maxiter=self.optimization_config.lbfgs_maxiter)

        # Logging in response to failures of optimization runs
        n_succeeded = sum(x is None for x in ret_infos)
        if n_succeeded < n_starts:
            log_msg = "[GaussianProcessLearningCurveModel.fit]\n"
            log_msg += ("{} of the {} restarts failed with the following exceptions:\n".format(
                n_starts - n_succeeded, n_starts))
            copy_params = {
                param.name: param.data()
                for param in self.likelihood.collect_params().values()}
            for i, ret_info in enumerate(ret_infos):
                if ret_info is not None:
                    log_msg += ("- Restart {}: Exception {}\n".format(
                        i, ret_info['type']))
                    log_msg += ("  Message: {}\n".format(ret_info['msg']))
                    log_msg += ("  Args: {}\n".format(ret_info['args']))
                    # Set parameters in order to print them. These are the
                    # parameters for which the evaluation failed
                    self._set_likelihood_params(ret_info['params'])
                    log_msg += ("  Params: " + str(self.get_params()))
                    logger.info(log_msg)
            # Restore parameters
            self._set_likelihood_params(copy_params)
            if n_succeeded == 0:
                logger.info("All restarts failed: Skipping hyperparameter fitting for now")
        # Recompute posterior state for new hyperparameters
        self._recompute_states(data)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(self, data: Dict):
        self._recompute_states(data)

    def _recompute_states(self, data: Dict):
        self._data_precomputations(data)
        self._states = [self.likelihood.get_posterior_state(data)]

    def predict(self, features_test):
        """
        Compute the posterior mean(s) and variance(s) for the points in features_test.

        :param features_test: Data matrix X_test
        :return: [(posterior_means, posterior_variances)]
        """
        return [state.predict(features_test) for state in self.states]

    def sample_marginals(self, features_test, num_samples=1):
        """
        Draws marginal samples from predictive distribution at n test points.
        We concatenate the samples for each state, returning a matrix of shape
        (n, num_samples * num_states), where num_states = len(self.states).

        :param features_test: Test input points, shape (n, d)
        :param num_samples: Number of samples
        :return: Samples with shape (n, num_samples * num_states)
        """
        samples_list = [
            state.sample_marginals(
                features_test, num_samples, random_state=self._random_state)
            for state in self.states]
        return anp.concatenate(samples_list, axis=-1)

    def get_params(self):
        return self.likelihood.get_params()

    def set_params(self, param_dict):
        self.likelihood.set_params(param_dict)

    def reset_params(self):
        """
        Reset hyperparameters to their initial values (or resample them).
        """
        # Note: The `init` parameter is a default sampler which is used only
        # for parameters which do not have initializers specified. Right now,
        # all our parameters have such initializers (constant in general),
        # so this is just to be safe (if `init` is not specified here, it
        # defaults to `np.random.uniform`, whose seed we do not control).
        self.likelihood.initialize(
            init=self._random_state.uniform, force_reinit=True)

    def _data_precomputations(self, data: Dict):
        """
        For some `res_model` types, precomputations on top of `data` are
        needed. This is done here, and the precomputed variables are appended
        to `data` as extra entries.

        :param data:
        """
        if self._use_precomputations and (
                self._states is None
                or not self._states[0].has_precomputations(data)):
            self.likelihood.data_precomputations(data)
