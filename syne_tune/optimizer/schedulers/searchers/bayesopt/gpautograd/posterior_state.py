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
from autograd import grad
from autograd.tracer import getval
from typing import Tuple, Optional, Dict, Callable
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_utils import (
    cholesky_computations,
    predict_posterior_marginals,
    sample_posterior_marginals,
    sample_posterior_joint,
    cholesky_update,
    negative_log_marginal_likelihood,
    sample_and_cholesky_update,
    KernelFunctionWithCovarianceScale,
)


class PosteriorState:
    """
    Interface for posterior state of Gaussian-linear model.
    """

    @property
    def num_data(self):
        raise NotImplementedError

    @property
    def num_features(self):
        raise NotImplementedError

    @property
    def num_fantasies(self):
        raise NotImplementedError

    def neg_log_likelihood(self) -> anp.ndarray:
        """
        :return: Negative log marginal likelihood
        """
        raise NotImplementedError

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes marginal statistics (means, variances) for a number of test
        features.

        :param test_features: Features for test configs
        :return: posterior_means, posterior_variances
        """
        raise NotImplementedError

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        See comments of `predict`.

        :param test_features: Input points for test configs
        :param num_samples: Number of samples
        :param random_state: PRNG
        :return: Marginal samples, (num_test, num_samples)
        """
        raise NotImplementedError

    def backward_gradient(
        self,
        input: np.ndarray,
        head_gradients: Dict[str, np.ndarray],
        mean_data: float,
        std_data: float,
    ) -> np.ndarray:
        """
        Implements SurrogateModel.backward_gradient, see comments there.
        This is for a single posterior state. If the SurrogateModel uses
        MCMC, have to call this for every sample.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See SurrogateModel.backward_gradient
        :param mean_data: Mean used to normalize targets
        :param std_data: Stddev used to normalize targets
        :return:
        """
        raise NotImplementedError


class PosteriorStateWithSampleJoint(PosteriorState):
    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        See comments of `predict`.

        :param test_features: Input points for test configs
        :param num_samples: Number of samples
        :param random_state: PRNG
        :return: Joint samples, (num_test, num_samples)
        """
        raise NotImplementedError


class GaussProcPosteriorState(PosteriorStateWithSampleJoint):
    """
    Represent posterior state for Gaussian process regression model.
    Note that members are immutable. If the posterior state is to be
    updated, a new object is created and returned.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray],
        mean: MeanFunction,
        kernel: KernelFunctionWithCovarianceScale,
        noise_variance: np.ndarray,
        debug_log: bool = False,
        **kwargs
    ):
        """
        If targets has m > 1 columns, they correspond to fantasy samples.

        If targets is None, this is an internal (copy) constructor, where
        kwargs contains chol_fact, pred_mat.

        `kernel` can be a tuple `(_kernel, covariance_scale)`, where
        `_kernel` is a `KernelFunction`, `covariance_scale` a scalar
        parameter. In this case, the kernel function is their product.

        :param features: Input points X, shape (n, d)
        :param targets: Targets Y, shape (n, m)
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X'), or tuple (see above)
        :param noise_variance: Noise variance sigsq, shape (1,)
        """
        self.mean = mean
        self.kernel = self._check_and_assign_kernel(kernel)
        self.noise_variance = anp.array(noise_variance, copy=True)
        if targets is not None:
            targets_shape = getval(targets.shape)
            targets = anp.reshape(targets, (targets_shape[0], -1))
            self.chol_fact, self.pred_mat = cholesky_computations(
                features=features,
                targets=targets,
                mean=mean,
                kernel=kernel,
                noise_variance=noise_variance,
                debug_log=debug_log,
            )
            self.features = anp.array(features, copy=True)
        else:
            # Internal (copy) constructor
            self.features = features
            self.chol_fact = kwargs["chol_fact"]
            self.pred_mat = kwargs["pred_mat"]

    @staticmethod
    def _check_and_assign_kernel(kernel: KernelFunctionWithCovarianceScale):
        if isinstance(kernel, tuple):
            assert len(kernel) == 2
            kernel, covariance_scale = kernel
            assert isinstance(kernel, KernelFunction)
            return kernel, anp.array(covariance_scale, copy=True)
        else:
            assert isinstance(kernel, KernelFunction)
            return kernel

    @property
    def num_data(self):
        return self.features.shape[0]

    @property
    def num_features(self):
        return self.features.shape[1]

    @property
    def num_fantasies(self):
        return self.pred_mat.shape[1]

    def _state_kwargs(self) -> dict:
        return {
            "features": self.features,
            "mean": self.mean,
            "kernel": self.kernel,
            "chol_fact": self.chol_fact,
            "pred_mat": self.pred_mat,
        }

    def neg_log_likelihood(self) -> anp.ndarray:
        """
        Works only if fantasy samples are not used (single targets vector).
        """
        critval = negative_log_marginal_likelihood(self.chol_fact, self.pred_mat)
        return critval

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return predict_posterior_marginals(
            **self._state_kwargs(), test_features=test_features
        )

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        if random_state is None:
            random_state = np.random
        return sample_posterior_marginals(
            **self._state_kwargs(),
            test_features=test_features,
            random_state=random_state,
            num_samples=num_samples
        )

    def backward_gradient(
        self,
        input: np.ndarray,
        head_gradients: Dict[str, np.ndarray],
        mean_data: float,
        std_data: float,
    ) -> np.ndarray:
        """
        Implements SurrogateModel.backward_gradient, see comments there.
        This is for a single posterior state. If the SurrogateModel uses
        MCMC, have to call this for every sample.

        The posterior represented here is based on normalized data, while
        the acquisition function is based on the de-normalized predictive
        distribution, which is why we need 'mean_data', 'std_data' here.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See SurrogateModel.backward_gradient
        :param mean_data: Mean used to normalize targets
        :param std_data: Stddev used to normalize targets
        :return:
        """

        def predict_func(test_feature_array):
            return self.predict(test_feature_array)

        return backward_gradient_given_predict(
            predict_func=predict_func,
            input=input,
            head_gradients=head_gradients,
            mean_data=mean_data,
            std_data=std_data,
        )

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        if random_state is None:
            random_state = np.random
        return sample_posterior_joint(
            **self._state_kwargs(),
            test_features=test_features,
            random_state=random_state,
            num_samples=num_samples
        )


def backward_gradient_given_predict(
    predict_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    input: np.ndarray,
    head_gradients: Dict[str, np.ndarray],
    mean_data: float,
    std_data: float,
) -> np.ndarray:
    """
    Implements SurrogateModel.backward_gradient, see comments there.
    This is for a single posterior state. If the SurrogateModel uses
    MCMC, have to call this for every sample.

    The posterior represented here is based on normalized data, while
    the acquisition function is based on the de-normalized predictive
    distribution, which is why we need 'mean_data', 'std_data' here.

    :param predict_func: Function mapping input x to mean, variance
    :param input: Single input point x, shape (d,)
    :param head_gradients: See SurrogateModel.backward_gradient
    :param mean_data: Mean used to normalize targets
    :param std_data: Stddev used to normalize targets
    :return:
    """
    test_feature = np.reshape(input, (1, -1))
    assert "mean" in head_gradients, "Need head_gradients['mean'] for backward_gradient"
    has_std = "std" in head_gradients

    def diff_test_feature(test_feature_array):
        norm_mean, norm_variance = predict_func(test_feature_array)
        # De-normalize, and variance -> stddev
        pred_mean = norm_mean * std_data + mean_data
        head_gradients_mean = anp.reshape(head_gradients["mean"], pred_mean.shape)
        # Added to mimic mxnet.autograd.backward
        pred_mean_sum = anp.sum(anp.multiply(pred_mean, head_gradients_mean))
        if has_std:
            pred_std = anp.sqrt(norm_variance) * std_data
            head_gradients_std = anp.reshape(head_gradients["std"], pred_std.shape)
            pred_std_sum = anp.sum(anp.multiply(pred_std, head_gradients_std))
        else:
            pred_std_sum = 0.0
        return pred_mean_sum + pred_std_sum

    test_feature_gradient = grad(diff_test_feature)
    return np.reshape(test_feature_gradient(test_feature), input.shape)


class IncrementalUpdateGPPosteriorState(GaussProcPosteriorState):
    """
    Extension of GaussProcPosteriorState which allows for incremental
    updating, given that a single data case is appended to the training
    set.

    In order to not mutate members,
    "the update method returns a new object."
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray],
        mean: MeanFunction,
        kernel: KernelFunctionWithCovarianceScale,
        noise_variance: np.ndarray,
        **kwargs
    ):
        super(IncrementalUpdateGPPosteriorState, self).__init__(
            features, targets, mean, kernel, noise_variance, **kwargs
        )

    def update(
        self, feature: np.ndarray, target: np.ndarray
    ) -> "IncrementalUpdateGPPosteriorState":
        """
        :param feature: Additional input xstar, shape (1, d)
        :param target: Additional target ystar, shape (1, m)
        :return: Posterior state for increased data set
        """
        feature = anp.reshape(feature, (1, -1))
        target = anp.reshape(target, (1, -1))
        assert (
            feature.shape[1] == self.features.shape[1]
        ), "feature.shape[1] = {} != {} = self.features.shape[1]".format(
            feature.shape[1], self.features.shape[1]
        )
        assert (
            target.shape[1] == self.pred_mat.shape[1]
        ), "target.shape[1] = {} != {} = self.pred_mat.shape[1]".format(
            target.shape[1], self.pred_mat.shape[1]
        )
        chol_fact_new, pred_mat_new = cholesky_update(
            **self._state_kwargs(),
            noise_variance=self.noise_variance,
            feature=feature,
            target=target
        )
        features_new = anp.concatenate([self.features, feature], axis=0)
        state_new = IncrementalUpdateGPPosteriorState(
            features=features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new,
        )
        return state_new

    def sample_and_update(
        self,
        feature: np.ndarray,
        mean_impute_mask=None,
        random_state: Optional[RandomState] = None,
    ) -> (np.ndarray, "IncrementalUpdateGPPosteriorState"):
        """
        Draw target(s), shape (1, m), from current posterior state, then update
        state based on these. The main computation of lvec is shared among the
        two.
        If mean_impute_mask is given, it is a boolean vector of size m (number
        columns of pred_mat). Columns j of target, where mean_impute_ mask[j]
        is true, are set to the predictive mean (instead of being sampled).

        :param feature: Additional input xstar, shape (1, d)
        :param mean_impute_mask: See above
        :param random_state: PRN generator
        :return: target, poster_state_new
        """
        feature = anp.reshape(feature, (1, -1))
        assert (
            feature.shape[1] == self.features.shape[1]
        ), "feature.shape[1] = {} != {} = self.features.shape[1]".format(
            feature.shape[1], self.features.shape[1]
        )
        if random_state is None:
            random_state = np.random
        chol_fact_new, pred_mat_new, features_new, target = sample_and_cholesky_update(
            **self._state_kwargs(),
            noise_variance=self.noise_variance,
            feature=feature,
            random_state=random_state,
            mean_impute_mask=mean_impute_mask
        )
        state_new = IncrementalUpdateGPPosteriorState(
            features=features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new,
        )
        return target, state_new

    def expand_fantasies(
        self, num_fantasies: int
    ) -> "IncrementalUpdateGPPosteriorState":
        """
        If this posterior has been created with a single targets vector,
        shape (n, 1), use this to duplicate this vector m = num_fantasies
        times. Call this method before fantasy targets are appended by
        update.

        :param num_fantasies: Number m of fantasy samples
        :return: New state with targets duplicated m times
        """
        assert num_fantasies > 1
        assert (
            self.pred_mat.shape[1] == 1
        ), "Method requires posterior without fantasy samples"
        pred_mat_new = anp.concatenate(([self.pred_mat] * num_fantasies), axis=1)
        return IncrementalUpdateGPPosteriorState(
            features=self.features,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=self.chol_fact,
            pred_mat=pred_mat_new,
        )
