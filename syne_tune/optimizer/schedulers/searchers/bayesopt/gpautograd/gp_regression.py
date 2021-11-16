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
from typing import Optional, List

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model \
    import GaussianProcessModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood \
    import MarginalLikelihood
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ScalarMeanFunction, MeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.optimization_utils \
    import apply_lbfgs_with_multiple_starts, create_lbfgs_arguments
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state \
    import GaussProcPosteriorState
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler

logger = logging.getLogger(__name__)


class GaussianProcessRegression(GaussianProcessModel):
    """
    Gaussian Process Regression

    Takes as input a mean function (which depends on X only) and a kernel
    function.

    :param kernel: Kernel function (for instance, a Matern52---note we cannot
        provide Matern52() as default argument since we need to provide with
        the dimension of points in X)
    :param mean: Mean function (which depends on X only)
    :param initial_noise_variance: Initial value for noise variance parameter
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values

    """
    def __init__(
            self, kernel: KernelFunction, mean: MeanFunction = None,
            initial_noise_variance: float = None,
            optimization_config: OptimizationConfig = None,
            random_seed=None, fit_reset_params: bool = True,
            test_intermediates: Optional[dict] = None):
        super().__init__(random_seed)
        if mean is None:
            mean = ScalarMeanFunction()
        if optimization_config is None:
            optimization_config = DEFAULT_OPTIMIZATION_CONFIG
        self._states = None
        self.fit_reset_params = fit_reset_params
        self.optimization_config = optimization_config
        self._test_intermediates = test_intermediates
        self.likelihood = MarginalLikelihood(
            kernel=kernel, mean=mean,
            initial_noise_variance=initial_noise_variance)
        self.reset_params()

    @property
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._states

    def fit(self, features, targets, profiler: SimpleProfiler = None):
        """
        Fit the parameters of the GP by optimizing the marginal likelihood,
        and set posterior states.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        :param features: data matrix X of size (n, d)
        :param targets: matrix of targets of size (n, 1)
        """
        features, targets = self._check_features_targets(features, targets)
        assert targets.shape[1] == 1, \
            "targets cannot be a matrix if parameters are to be fit"

        if self.fit_reset_params:
            self.reset_params()
        mean_function = self.likelihood.mean
        if isinstance(mean_function, ScalarMeanFunction):
            mean_function.set_mean_value(np.mean(targets))
        if profiler is not None:
            profiler.start('fithyperpars')
        n_starts = self.optimization_config.n_starts
        ret_infos = apply_lbfgs_with_multiple_starts(
            *create_lbfgs_arguments(
                criterion=self.likelihood,
                crit_args=[features, targets],
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
            log_msg = "[GaussianProcessRegression.fit]\n"
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
        self._recompute_states(features, targets, profiler=profiler)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(
            self, features, targets, profiler: SimpleProfiler = None):
        """
        We allow targets to be a matrix with m>1 columns, which is useful to support
        batch decisions by fantasizing.
        """
        features, targets = self._check_features_targets(features, targets)
        self._recompute_states(features, targets, profiler=profiler)

    def _recompute_states(
            self, features, targets, profiler: SimpleProfiler = None):
        if profiler is not None:
            profiler.start('posterstate')
        self._states = [GaussProcPosteriorState(
            features, targets, self.likelihood.mean, self.likelihood.kernel,
            self.likelihood.get_noise_variance(as_ndarray=True),
            debug_log=(self._test_intermediates is not None),
            test_intermediates=self._test_intermediates)]
        if profiler is not None:
            profiler.stop('posterstate')
    
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
