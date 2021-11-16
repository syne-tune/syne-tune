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
    import MarginalLikelihood
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import ISSModelParameters
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.posterior_state \
    import GaussProcISSMPosteriorState
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


# TODO: Method for joint sampling of curves
class GaussianProcessISSModel(object):
    """
    Represents joint Gaussian model of learning curves over a number of
    configurations. The model is the sum of a Gaussian process model for
    function values at r_max and independent Gaussian linear innovation state
    space models (ISSMs) of a particular power law decay form. Importantly,
    inference scales cubically only in the number of configurations, not in the
    number of observations.

    Details about GP-ISSM are contained in an internal report. Details about
    ISSMs in general are found in

        Hyndman, R. and Koehler, A. and Ord, J. and Snyder, R.
        Forecasting with Exponential Smoothing: The State Space Approach
        Springer, 2008

    :param kernel: Kernel function k(X, X')
    :param iss_model: ISS model
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
            self, kernel: KernelFunction, iss_model: ISSModelParameters,
            mean: MeanFunction = None, initial_noise_variance: float = None,
            optimization_config: OptimizationConfig = None, random_seed=None,
            fit_reset_params: bool = True):

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
        self.likelihood = MarginalLikelihood(
            kernel=kernel,
            iss_model=iss_model,
            mean=mean,
            initial_noise_variance=initial_noise_variance)
        self.reset_params()

    @property
    def states(self) -> Optional[List[GaussProcISSMPosteriorState]]:
        return self._states

    def fit(self, data: Dict, profiler: SimpleProfiler = None):
        """
        Fit the (hyper)parameters of the model by optimizing the marginal
        likelihood, and set posterior states. `data` is obtained from
        `TuningJobState` by `issm.prepare_data`.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        :param data: Input points (features, configs), targets
        """
        if self.fit_reset_params:
            self.reset_params()
        if profiler is not None:
            profiler.start('fithyperpars')
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
        if profiler is not None:
            profiler.stop('fithyperpars')

        # Logging in response to failures of optimization runs
        n_succeeded = sum(x is None for x in ret_infos)
        if n_succeeded < n_starts:
            log_msg = "[GaussianProcessISSModel.fit]\n"
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
        self._recompute_states(data, profiler=profiler)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(self, data: Dict, profiler: SimpleProfiler = None):
        assert self._states is not None, \
            "Have to call 'fit' at least once to set hyperparameter values"
        self._recompute_states(data, profiler=profiler)

    def _recompute_states(self, data: Dict, profiler: SimpleProfiler = None):
        if profiler is not None:
            profiler.start('posterstate')
        self._states = [GaussProcISSMPosteriorState(
            data=data,
            mean=self.likelihood.mean,
            kernel=self.likelihood.kernel,
            iss_model=self.likelihood.iss_model,
            noise_variance=self.likelihood.get_noise_variance())]
        if profiler is not None:
            profiler.stop('posterstate')

    def predict(self, features_test):
        """
        Compute the posterior mean(s) and variance(s) for the points in features_test.

        :param features_test: Data matrix X_test
        :return: [(posterior_means, posterior_variances)]
        """
        predictions = []
        for state in self.states:
            post_means, post_vars = state.predict(features_test)
            predictions.append((post_means, post_vars))
        return predictions

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
