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
from typing import Dict, Tuple, Optional, Callable, Set
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
    NoiseVariance,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.utils import (
    ExtendFeaturesByResourceMixin,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    backward_gradient_given_predict,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_utils import (
    KernelFunctionWithCovarianceScale,
)


def assert_ensemble_distribution(
    distribution: Dict[int, float], all_resources: Set[int]
):
    assert set(distribution.keys()).issubset(all_resources), (
        f"distribution.keys() = {set(distribution.keys())} must be subset "
        f"of {all_resources}"
    )
    assert all(
        x > 0 for x in distribution.values()
    ), f"Values {distribution.values()} must be positive"


def _sample_hypertune_common(
    ensemble_distribution: Dict[int, float],
    sample_func: Callable[[int, int], np.ndarray],
    num_samples: int,
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    if random_state is None:
        random_state = np.random
    supp_resources, theta = zip(*list(ensemble_distribution.items()))
    num_per_resource = random_state.multinomial(
        n=num_samples, pvals=theta, size=1
    ).reshape((-1,))
    all_samples = []
    for n_samples, resource in zip(num_per_resource, supp_resources):
        if n_samples > 0:
            all_samples.append(sample_func(resource, n_samples))
    samples = np.concatenate(all_samples, axis=-1)
    ind = random_state.permutation(num_samples)
    return np.take(samples, ind, axis=-1)


def _assert_features_shape(test_features: np.ndarray, num_features: int) -> int:
    dimension = num_features - 1
    assert (
        test_features.ndim == 2 and dimension <= test_features.shape[1] <= dimension + 1
    ), (
        f"test_features.shape = {test_features.shape}, must be "
        f"(*, {dimension}) or (*, {dimension + 1})"
    )
    return dimension


class HyperTuneIndependentGPPosteriorState(IndependentGPPerResourcePosteriorState):
    """
    Special case of :class:`IndependentGPPerResourcePosteriorState`, where
    methods `predict`, `backward_gradient`, `sample_marginals`, `sample_joint`
    are over a random function :math:`f_{MF}(x)`, obtained by first sampling the
    resource level :math:`r \\sim [\\theta_r]`, then use
    :math:`f_{MF}(x) = f(x, r)`. Predictive means and variances are:

    ..math::
        \\mu_{MF}(x) = \\sum_r \\theta_r \\mu(x, r)
        \\sigma_{MF}^2(x) = \\sum_r \\theta_r^2 \\sigma_{MF}^2(x, r)

    Here, :math:`[\\theta_k]` is a distribution over a subset of rung levels.

    Note: This posterior state is unusual, in that `sample_marginals`,
    `sample_joint` have to work both with (a) extended inputs (x, r) and (b)
    non-extended inputs x. For case (a), they behave like the superclass
    methods, this is needed to support fitting model parameters, for example
    for drawing fantasy samples. For case (b), they use the ensemble
    distribution detailed above, which supports optimizing the acquisition
    function.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        covariance_scale: Dict[int, np.ndarray],
        noise_variance: NoiseVariance,
        resource_attr_range: Tuple[int, int],
        ensemble_distribution: Dict[int, float],
        debug_log: bool = False,
    ):
        """
        `ensemble_distribution` contains non-zero entries of the distribution
        :math:`[\\theta_k]`. All resource levels supported there must have
        sufficient data in order to allow for predictions.
        """
        super().__init__(
            features=features,
            targets=targets,
            kernel=kernel,
            mean=mean,
            covariance_scale=covariance_scale,
            noise_variance=noise_variance,
            resource_attr_range=resource_attr_range,
            debug_log=debug_log,
        )
        assert_ensemble_distribution(ensemble_distribution, set(mean.keys()))
        self.ensemble_distribution = ensemble_distribution

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means, variances = 0, 0
        for resource, theta in self.ensemble_distribution.items():
            _means, _variances = self._states[resource].predict(test_features)
            means = _means * theta + means
            variances = _variances * (theta * theta) + variances
        return means, variances

    def _sample_internal_hypertune(
        self,
        sample_func: Callable[[int, int], np.ndarray],
        num_samples: int,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        return _sample_hypertune_common(
            ensemble_distribution=self.ensemble_distribution,
            sample_func=sample_func,
            num_samples=num_samples,
            random_state=random_state,
        )

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        If `test_features` are non-extended features (no resource attribute),
        we sample from the ensemble predictive distribution. Otherwise, we
        call the superclass method.
        """
        dimension = _assert_features_shape(test_features, self.num_features)
        if test_features.shape[1] == dimension:

            def sample_func(resource: int, n_samples: int):
                return self._states[resource].sample_marginals(
                    test_features,
                    num_samples=n_samples,
                    random_state=random_state,
                )

            return self._sample_internal_hypertune(
                sample_func=sample_func,
                num_samples=num_samples,
                random_state=random_state,
            )
        else:
            return super().sample_marginals(
                test_features=test_features,
                num_samples=num_samples,
                random_state=random_state,
            )

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        If `test_features` are non-extended features (no resource attribute),
        we sample from the ensemble predictive distribution. Otherwise, we
        call the superclass method.
        """
        dimension = _assert_features_shape(test_features, self.num_features)
        if test_features.shape[1] == dimension:

            def sample_func(resource: int, n_samples: int):
                return self._states[resource].sample_joint(
                    test_features,
                    num_samples=n_samples,
                    random_state=random_state,
                )

            return self._sample_internal_hypertune(
                sample_func=sample_func,
                num_samples=num_samples,
                random_state=random_state,
            )
        else:
            return super().sample_joint(
                test_features=test_features,
                num_samples=num_samples,
                random_state=random_state,
            )

    def backward_gradient(
        self,
        input: np.ndarray,
        head_gradients: Dict[str, np.ndarray],
        mean_data: float,
        std_data: float,
    ) -> np.ndarray:
        def predict_func(test_feature_array):
            return self.predict(test_feature_array)

        return backward_gradient_given_predict(
            predict_func=predict_func,
            input=input,
            head_gradients=head_gradients,
            mean_data=mean_data,
            std_data=std_data,
        )


class HyperTuneJointGPPosteriorState(GaussProcPosteriorState):
    """
    Special case of :class:`GaussProcPosteriorState`, where methods `predict`,
    `backward_gradient`, `sample_marginals`, `sample_joint` are over a random
    function :math:`f_{MF}(x)`, obtained by first sampling the resource level
    :math:`r \\sim [\\theta_r]`, then use :math:`f_{MF}(x) = f(x, r)`.
    Predictive means and variances are:

    ..math::
        \\mu_{MF}(x) = \\sum_r \\theta_r \\mu(x, r)
        \\sigma_{MF}^2(x) = \\sum_r \\theta_r^2 \\sigma_{MF}^2(x, r)

    Here, :math:`[\\theta_k]` is a distribution over a subset of rung levels.

    Note: This posterior state is unusual, in that `sample_marginals`,
    `sample_joint` have to work both with (a) extended inputs (x, r) and (b)
    non-extended inputs x. For case (a), they behave like the superclass
    methods, this is needed to support fitting model parameters, for example
    for drawing fantasy samples. For case (b), they use the ensemble
    distribution detailed above, which supports optimizing the acquisition
    function.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        mean: MeanFunction,
        kernel: KernelFunctionWithCovarianceScale,
        noise_variance: np.ndarray,
        resource_attr_range: Tuple[int, int],
        ensemble_distribution: Dict[int, float],
        debug_log: bool = False,
    ):
        """
        `ensemble_distribution` contains non-zero entries of the distribution
        :math:`[\\theta_k]`. All resource levels supported there must have
        sufficient data in order to allow for predictions.
        """
        super().__init__(
            features=features,
            targets=targets,
            mean=mean,
            kernel=kernel,
            noise_variance=noise_variance,
            debug_log=debug_log,
        )
        self.ensemble_distribution = ensemble_distribution
        self._resource_attr_range = resource_attr_range

    def _extend_features_by_resource(
        self, test_features: np.ndarray, resource: int
    ) -> np.ndarray:
        helper = ExtendFeaturesByResourceMixin(
            resource=resource, resource_attr_range=self._resource_attr_range
        )
        return helper.extend_features_by_resource(test_features)

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means, variances = 0, 0
        for resource, theta in self.ensemble_distribution.items():
            features_ext = self._extend_features_by_resource(test_features, resource)
            _means, _variances = super().predict(features_ext)
            means = _means * theta + means
            variances = _variances * (theta * theta) + variances
        return means, variances

    def _sample_internal_hypertune(
        self,
        sample_func: Callable[[int, int], np.ndarray],
        num_samples: int,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        return _sample_hypertune_common(
            ensemble_distribution=self.ensemble_distribution,
            sample_func=sample_func,
            num_samples=num_samples,
            random_state=random_state,
        )

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        If `test_features` are non-extended features (no resource attribute),
        we sample from the ensemble predictive distribution. Otherwise, we
        call the superclass method.
        """
        dimension = _assert_features_shape(test_features, self.num_features)
        if test_features.shape[1] == dimension:

            def sample_func(resource: int, n_samples: int):
                features_ext = self._extend_features_by_resource(
                    test_features, resource
                )
                return super().sample_marginals(
                    features_ext, num_samples=n_samples, random_state=random_state
                )

            return self._sample_internal_hypertune(
                sample_func=sample_func,
                num_samples=num_samples,
                random_state=random_state,
            )
        else:
            return super().sample_marginals(
                test_features=test_features,
                num_samples=num_samples,
                random_state=random_state,
            )

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        """
        If `test_features` are non-extended features (no resource attribute),
        we sample from the ensemble predictive distribution. Otherwise, we
        call the superclass method.
        """
        dimension = _assert_features_shape(test_features, self.num_features)
        if test_features.shape[1] == dimension:

            def sample_func(resource: int, n_samples: int):
                features_ext = self._extend_features_by_resource(
                    test_features, resource
                )
                return super().sample_joint(
                    features_ext, num_samples=n_samples, random_state=random_state
                )

            return self._sample_internal_hypertune(
                sample_func=sample_func,
                num_samples=num_samples,
                random_state=random_state,
            )
        else:
            return super().sample_joint(
                test_features=test_features,
                num_samples=num_samples,
                random_state=random_state,
            )

    def backward_gradient(
        self,
        input: np.ndarray,
        head_gradients: Dict[str, np.ndarray],
        mean_data: float,
        std_data: float,
    ) -> np.ndarray:
        def predict_func(test_feature_array):
            return self.predict(test_feature_array)

        return backward_gradient_given_predict(
            predict_func=predict_func,
            input=input,
            head_gradients=head_gradients,
            mean_data=mean_data,
            std_data=std_data,
        )
