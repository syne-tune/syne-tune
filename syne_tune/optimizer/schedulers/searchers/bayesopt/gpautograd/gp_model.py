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
from abc import ABC, abstractmethod
from typing import List, Optional

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import DATA_TYPE
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state \
    import GaussProcPosteriorState


class GaussianProcessModel(ABC):
    def __init__(self, random_seed=None):
        if random_seed is None:
            random_seed = 31415927
        self._random_state = np.random.RandomState(random_seed)

    @property
    @abstractmethod
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        pass

    @abstractmethod
    def fit(self, features: anp.array, targets: anp.array):
        """Train GP on the data and set a list of posterior states to be used by predict function"""
        pass

    @abstractmethod
    def recompute_states(self, features: anp.array, targets: anp.array):
        """Fixing GP hyperparameters and recompute the list of posterior states based on features and targets"""
        pass

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
        assert features.shape[0] == targets.shape[0], \
            f"features and targets should have the same number of points " +\
            f"(received {features.shape[0]} and {targets.shape[0]})"
        return features, targets

    def predict(self, features_test):
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

    def _assert_check_xtest(self, features_test):
        assert self.states is not None, \
            "Posterior state does not exist (run 'fit')"
        features_test = self._check_and_format_input(features_test)
        assert features_test.shape[1] == self.states[0].num_features,\
            f"features_test and features should have the same number of " + \
            f"columns (received {features_test.shape[1]}, expected " + \
            f"{self.states[0].num_features})"
        return features_test

    def multiple_targets(self):
        """
        :return: Posterior state based on multiple (fantasized) target
        """
        assert self.states is not None, \
            "Posterior state does not exist (run 'fit')"
        return self.states[0].num_fantasies > 1

    def sample_marginals(self, features_test, num_samples=1):
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
                features_test, num_samples, random_state=self._random_state)
            for state in self.states]
        return _concatenate_samples(samples_list)

    def sample_joint(self, features_test, num_samples=1):
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
        samples_list = [
            state.sample_joint(
                features_test, num_samples, random_state=self._random_state)
            for state in self.states]
        return _concatenate_samples(samples_list)


def _concatenate_samples(samples_list: List[anp.ndarray]) -> anp.ndarray:
    return anp.concatenate(samples_list, axis=-1)
