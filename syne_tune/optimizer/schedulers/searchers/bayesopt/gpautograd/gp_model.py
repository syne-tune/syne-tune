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
import numpy as np
import autograd.numpy as anp
from autograd.builtins import isinstance
from typing import List, Optional
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    DATA_TYPE,
    OptimizationConfig,
    DEFAULT_OPTIMIZATION_CONFIG,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
    PosteriorStateWithSampleJoint,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.optimization_utils import (
    apply_lbfgs_with_multiple_starts,
    create_lbfgs_arguments,
)

logger = logging.getLogger(__name__)


class GaussianProcessModel:
    """
    Base class for Gaussian-linear models which support parameter fitting and
    prediction.
    """

    def __init__(self, random_seed=None):
        if random_seed is None:
            random_seed = 31415927
        self._random_state = np.random.RandomState(random_seed)

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random_state

    @property
    def states(self) -> Optional[List[PosteriorState]]:
        """
        :return: Current posterior states (one per MCMC sample; just a single
            state if model parameters are optimized)
        """
        raise NotImplementedError

    def fit(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        """
        Adjust model parameters based on training data `data`. Can be done via
        optimization or MCMC sampling. The posterior states are computed at the
        end as well.

        :param data: Training data
        :param profiler: Used for profiling
        """
        raise NotImplementedError

    def recompute_states(self, data: dict):
        """
        Recomputes posterior states for current model parameters.

        :param data: Training data
        """
        raise NotImplementedError

    @staticmethod
    def _check_and_format_input(u):
        """
        Check and massage the input to conform with the numerical type and context

        :param u: some np.ndarray
        """
        assert isinstance(u, anp.ndarray)

        if u.ndim == 1:
            u = anp.reshape(u, (-1, 1))
        if u.dtype != DATA_TYPE:
            return anp.array(u, dtype=DATA_TYPE)
        else:
            return u

    @staticmethod
    def _check_features_targets(features, targets):
        features = GaussianProcessModel._check_and_format_input(features)
        targets = GaussianProcessModel._check_and_format_input(targets)
        assert features.shape[0] == targets.shape[0], (
            f"features and targets should have the same number of points "
            + f"(received {features.shape[0]} and {targets.shape[0]})"
        )
        return features, targets

    def predict(self, features_test: np.ndarray):
        """
        Compute the posterior mean(s) and variance(s) for the points in features_test.
        If the posterior state is based on m target vectors, a (n, m) matrix is returned for posterior means.

        :param features_test: Data matrix X_test of size (n, d) (type np.ndarray) for which n predictions are made
        :return: posterior_means, posterior_variances
        """
        features_test = self._assert_check_xtest(features_test)

        predictions = []
        for state in self.states:
            post_means, post_vars = state.predict(features_test)
            # Just to make sure the return shapes are the same as before:
            if post_means.shape[1] == 1:
                post_means = anp.reshape(post_means, (-1,))
            predictions.append((post_means, post_vars))
        return predictions

    def _assert_check_xtest(self, features_test: np.ndarray):
        assert self.states is not None, "Posterior state does not exist (run 'fit')"
        features_test = self._check_and_format_input(features_test)
        return features_test

    def multiple_targets(self):
        """
        :return: Posterior state based on multiple (fantasized) target
        """
        assert self.states is not None, "Posterior state does not exist (run 'fit')"
        return self.states[0].num_fantasies > 1

    def sample_marginals(self, features_test: np.ndarray, num_samples: int = 1):
        """
        Draws marginal samples from predictive distribution at n test points.
        Notice we concat the samples for each state. Let n_states = len(self._states)

        If the posterior state is based on m > 1 target vectors, a
        (n, m, num_samples * n_states) tensor is returned, for m == 1 we return a
        (n, num_samples * n_states) matrix.

        :param features_test: Test input points, shape (n, d)
        :param num_samples: Number of samples
        :return: Samples with shape (n, num_samples * n_states) or
            (n, m, num_samples * n_states) if m > 1
        """
        features_test = self._assert_check_xtest(features_test)
        samples_list = [
            state.sample_marginals(
                features_test, num_samples, random_state=self._random_state
            )
            for state in self.states
        ]
        return _concatenate_samples(samples_list)

    def sample_joint(self, features_test: np.ndarray, num_samples: int = 1):
        """
        Draws joint samples from predictive distribution at n test points.
        This scales cubically with n.
        the posterior state must be based on a single target vector
        (m > 1 is not supported).

        :param features_test: Test input points, shape (n, d)
        :param num_samples: Number of samples
        :return: Samples, shape (n, num_samples)
        """
        features_test = self._assert_check_xtest(features_test)
        assert isinstance(
            self.states[0], PosteriorStateWithSampleJoint
        ), "Implemented only for joint Gaussian process models"
        samples_list = [
            state.sample_joint(
                features_test, num_samples, random_state=self._random_state
            )
            for state in self.states
        ]
        return _concatenate_samples(samples_list)


def _concatenate_samples(samples_list: List[anp.ndarray]) -> anp.ndarray:
    return anp.concatenate(samples_list, axis=-1)


class GaussianProcessOptimizeModel(GaussianProcessModel):
    """
    Base class for models where parameters are fit by maximizing the marginal
    likelihood.
    """

    def __init__(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        super().__init__(random_seed)
        if optimization_config is None:
            optimization_config = DEFAULT_OPTIMIZATION_CONFIG
        self._states = None
        self.fit_reset_params = fit_reset_params
        self.optimization_config = optimization_config

    @property
    def states(self) -> Optional[List[PosteriorState]]:
        return self._states

    @property
    def likelihood(self) -> MarginalLikelihood:
        raise NotImplementedError

    def fit(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        """
        Fit the model parameters by optimizing the marginal likelihood,
        and set posterior states.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        :param data: Input data
        :param profiler: Profiler, optional
        """
        self.likelihood.on_fit_start(data, profiler)
        if self.fit_reset_params:
            self.reset_params()
        n_starts = self.optimization_config.n_starts
        ret_infos = apply_lbfgs_with_multiple_starts(
            *create_lbfgs_arguments(
                criterion=self.likelihood,
                crit_args=[data],
                verbose=self.optimization_config.verbose,
            ),
            bounds=self.likelihood.box_constraints_internal(),
            random_state=self._random_state,
            n_starts=n_starts,
            tol=self.optimization_config.lbfgs_tol,
            maxiter=self.optimization_config.lbfgs_maxiter,
        )

        # Logging in response to failures of optimization runs
        n_succeeded = sum(x is None for x in ret_infos)
        if n_succeeded < n_starts:
            log_msg = "[GaussianProcessOptimizeModel.fit]\n"
            log_msg += (
                "{} of the {} restarts failed with the following exceptions:\n".format(
                    n_starts - n_succeeded, n_starts
                )
            )
            copy_params = {
                param.name: param.data()
                for param in self.likelihood.collect_params().values()
            }
            for i, ret_info in enumerate(ret_infos):
                if ret_info is not None:
                    log_msg += "- Restart {}: Exception {}\n".format(
                        i, ret_info["type"]
                    )
                    log_msg += "  Message: {}\n".format(ret_info["msg"])
                    log_msg += "  Args: {}\n".format(ret_info["args"])
                    # Set parameters in order to print them. These are the
                    # parameters for which the evaluation failed
                    self._set_likelihood_params(ret_info["params"])
                    log_msg += "  Params: " + str(self.get_params())
                    logger.info(log_msg)
            # Restore parameters
            self._set_likelihood_params(copy_params)
            if n_succeeded == 0:
                logger.info(
                    "All restarts failed: Skipping hyperparameter fitting for now"
                )
        # Recompute posterior state for new hyperparameters
        self._recompute_states(data)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(self, data: dict):
        self._recompute_states(data)

    def _recompute_states(self, data: dict):
        self.likelihood.data_precomputations(data)
        self._states = [self.likelihood.get_posterior_state(data)]

    def get_params(self):
        return self.likelihood.get_params()

    def set_params(self, param_dict):
        self.likelihood.set_params(param_dict)

    def reset_params(self):
        """
        Reset hyperparameters to their initial values (or resample them).
        """
        self.likelihood.reset_params(self._random_state)
