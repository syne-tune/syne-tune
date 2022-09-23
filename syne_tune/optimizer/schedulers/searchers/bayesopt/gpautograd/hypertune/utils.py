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
from autograd.tracer import getval
from typing import Optional, Tuple, List, Union, Dict, Callable
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_impl import (
    decode_extended_features,
    HyperparameterRangeInteger,
)
from syne_tune.optimizer.schedulers.searchers.utils.scaling import (
    LinearScaling,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    GaussProcPosteriorState,
    PosteriorStateWithSampleJoint,
    backward_gradient_given_predict,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (
    KernelFunction,
)


class ExtendFeaturesByResourceMixin:
    def __init__(self, resource: int, resource_attr_range: Tuple[int, int]):
        hp_range = HyperparameterRangeInteger(
            name="resource",
            lower_bound=resource_attr_range[0],
            upper_bound=resource_attr_range[1],
            scaling=LinearScaling(),
        )
        self._resource_encoded = hp_range.to_ndarray(resource).item()

    def extend_features_by_resource(self, test_features: np.ndarray) -> np.ndarray:
        shape = (getval(test_features.shape[0]), 1)
        extra_col = anp.full(shape, self._resource_encoded)
        return anp.concatenate((test_features, extra_col), axis=1)


class PosteriorStateClampedResource(
    PosteriorStateWithSampleJoint, ExtendFeaturesByResourceMixin
):
    """
    Converts posterior state of :class:`GaussPosteriorStateWithSampleJoint`
    over extended inputs into posterior state over non-extended inputs, where
    the resource attribute is clamped to a fixed value.
    """

    def __init__(
        self,
        poster_state_extended: PosteriorStateWithSampleJoint,
        resource: int,
        resource_attr_range: Tuple[int, int],
    ):
        ExtendFeaturesByResourceMixin.__init__(self, resource, resource_attr_range)
        self._poster_state_extended = poster_state_extended

    @property
    def num_data(self):
        return self._poster_state_extended.num_data

    @property
    def num_features(self):
        return self._poster_state_extended.num_features - 1

    @property
    def num_fantasies(self):
        return self._poster_state_extended.num_fantasies

    def neg_log_likelihood(self) -> anp.ndarray:
        return self._poster_state_extended.neg_log_likelihood()

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._poster_state_extended.predict(
            self.extend_features_by_resource(test_features)
        )

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        return self._poster_state_extended.sample_marginals(
            test_features=self.extend_features_by_resource(test_features),
            num_samples=num_samples,
            random_state=random_state,
        )

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        return self._poster_state_extended.sample_joint(
            test_features=self.extend_features_by_resource(test_features),
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


class MeanFunctionClampedResource(MeanFunction, ExtendFeaturesByResourceMixin):
    def __init__(
        self,
        mean_extended: MeanFunction,
        resource: int,
        resource_attr_range: Tuple[int, int],
        **kwargs,
    ):
        MeanFunction.__init__(self, **kwargs)
        ExtendFeaturesByResourceMixin.__init__(self, resource, resource_attr_range)
        self._mean_extended = mean_extended

    def param_encoding_pairs(self):
        return self._mean_extended.param_encoding_pairs()

    def get_params(self):
        return self._mean_extended.get_params()

    def set_params(self, param_dict):
        self._mean_extended.set_params(param_dict)

    def forward(self, X):
        return self._mean_extended.forward(self.extend_features_by_resource(X))


class KernelFunctionClampedResource(KernelFunction, ExtendFeaturesByResourceMixin):
    def __init__(
        self,
        kernel_extended: KernelFunction,
        resource: int,
        resource_attr_range: Tuple[int, int],
        **kwargs,
    ):
        KernelFunction.__init__(self, dimension=kernel_extended.dimension - 1, **kwargs)
        ExtendFeaturesByResourceMixin.__init__(self, resource, resource_attr_range)
        self._kernel_extended = kernel_extended

    def param_encoding_pairs(self):
        return self._kernel_extended.param_encoding_pairs()

    def get_params(self):
        return self._kernel_extended.get_params()

    def set_params(self, param_dict):
        self._kernel_extended.set_params(param_dict)

    def diagonal(self, X):
        return self._kernel_extended.diagonal(self.extend_features_by_resource(X))

    def diagonal_depends_on_X(self):
        return self._kernel_extended.diagonal_depends_on_X()

    def forward(self, X1, X2):
        X1_ext = self.extend_features_by_resource(X1)
        if X2 is X1:
            X2_ext = X1_ext
        else:
            X2_ext = self.extend_features_by_resource(X2)
        return self._kernel_extended.forward(X1_ext, X2_ext)


class GaussProcPosteriorStateAndRungLevels(PosteriorStateWithSampleJoint):
    def __init__(
        self,
        poster_state: GaussProcPosteriorState,
        rung_levels: List[int],
    ):
        self._poster_state = poster_state
        self._rung_levels = rung_levels

    @property
    def poster_state(self) -> GaussProcPosteriorState:
        return self._poster_state

    @property
    def num_data(self):
        return self._poster_state.num_data

    @property
    def num_features(self):
        return self._poster_state.num_features

    @property
    def num_fantasies(self):
        return self._poster_state.num_fantasies

    def neg_log_likelihood(self) -> anp.ndarray:
        return self._poster_state.neg_log_likelihood()

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._poster_state.predict(test_features)

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        return self._poster_state.sample_marginals(
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
        return self._poster_state.sample_joint(
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
        return self._poster_state.backward_gradient(
            input=input,
            head_gradients=head_gradients,
            mean_data=mean_data,
            std_data=std_data,
        )

    @property
    def rung_levels(self) -> List[int]:
        return self._rung_levels


PerResourcePosteriorState = Union[
    IndependentGPPerResourcePosteriorState,
    GaussProcPosteriorStateAndRungLevels,
]


def _posterior_state_for_rung_level(
    poster_state: PerResourcePosteriorState,
    resource: int,
    resource_attr_range: Tuple[int, int],
) -> Union[GaussProcPosteriorState, PosteriorStateClampedResource]:
    if isinstance(poster_state, IndependentGPPerResourcePosteriorState):
        return poster_state.state(resource)
    else:
        return PosteriorStateClampedResource(
            poster_state_extended=poster_state,
            resource=resource,
            resource_attr_range=resource_attr_range,
        )


def hypertune_ranking_losses(
    poster_state: PerResourcePosteriorState,
    data: dict,
    num_samples: int,
    resource_attr_range: Tuple[int, int],
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    """
    Samples ranking loss values as defined in the Hyper-Tune paper. We return a
    matrix of size `(num_supp_levels, num_samples)`, where
    `num_supp_levels <= poster_state.rung_levels` is the number of rung levels
    supported by at least 6 labeled datapoints.

    The loss values depend on the cases in `data` at the level
    `poster_state.rung_levels[num_supp_levels - 1]`. We must have
    `num_supp_levels >= 2`.

    Loss values at this highest supported level are estimated by
    cross-validation (so the data at this level is split into training and
    test, where the training part is used to obtain the posterior state). The
    number of CV folds is `<= 5`, and such that each fold has at least two
    points.

    :param poster_state: Posterior state over rung levels
    :param data: Training data
    :param num_samples: Number of independent loss samples
    :param resource_attr_range: (r_min, r_max)
    :param random_state: PRNG state
    :return: See above
    """
    independent_models = isinstance(
        poster_state, IndependentGPPerResourcePosteriorState
    )
    if not independent_models:
        assert isinstance(poster_state, GaussProcPosteriorStateAndRungLevels), (
            "poster_state needs to be IndependentGPPerResourcePosteriorState "
            "or GaussProcPosteriorStateAndRungLevels"
        )
    rung_levels = poster_state.rung_levels
    (
        num_supp_levels,
        data_max_resource,
    ) = number_supported_levels_and_data_highest_level(
        rung_levels=rung_levels,
        data=data,
        resource_attr_range=resource_attr_range,
    )
    assert (
        num_supp_levels > 1
    ), "Need to have at least 6 labeled datapoints at 2nd lowest rung level"
    max_resource = rung_levels[num_supp_levels - 1]
    loss_values = np.zeros((num_supp_levels, num_samples))
    # All loss values except for maximum rung (which is special)
    common_kwargs = dict(
        data_max_resource=data_max_resource,
        num_samples=num_samples,
        random_state=random_state,
    )
    for pos, resource in enumerate(rung_levels[: (num_supp_levels - 1)]):
        loss_values[pos] = _losses_for_rung(
            poster_state=_posterior_state_for_rung_level(
                poster_state, resource, resource_attr_range
            ),
            **common_kwargs,
        )

    # Loss values for maximum rung: Five-fold cross-validation
    if independent_models:
        poster_state_max_resource = poster_state.state(max_resource)
        mean_max_resource = poster_state_max_resource.mean
        kernel_max_resource = poster_state_max_resource.kernel
        noise_variance = poster_state_max_resource.noise_variance
    else:
        poster_state_int = poster_state.poster_state
        mean_max_resource = MeanFunctionClampedResource(
            mean_extended=poster_state_int.mean,
            resource=max_resource,
            resource_attr_range=resource_attr_range,
        )
        kernel_max_resource = KernelFunctionClampedResource(
            kernel_extended=poster_state_int.kernel,
            resource=max_resource,
            resource_attr_range=resource_attr_range,
        )
        noise_variance = poster_state_int.noise_variance

    def poster_state_for_fold(
        features: np.ndarray, targets: np.ndarray
    ) -> PosteriorStateWithSampleJoint:
        return GaussProcPosteriorState(
            features=features,
            targets=targets,
            mean=mean_max_resource,
            kernel=kernel_max_resource,
            noise_variance=noise_variance,
        )

    loss_values[-1] = _losses_for_maximum_rung_by_cross_validation(
        poster_state_for_fold=poster_state_for_fold, **common_kwargs
    )
    return loss_values


def number_supported_levels_and_data_highest_level(
    rung_levels: List[int],
    data: dict,
    resource_attr_range: Tuple[int, int],
) -> Tuple[int, dict]:
    """
    Finds `num_supp_levels` as maximum such that
    rung levels up to there have >= 6 labeled datapoints. The set
    of labeled datapoints of level `num_supp_levels - 1` is
    returned as well.

    If 'num_supp_levels == 1`, no level except for the lowest
    has >= 6 datapoints. In this case, `data_max_resource` returned
    is invalid.

    :param rung_levels: Rung levels
    :param data: Training data (only data at highest level is used)
    :param resource_attr_range: `(r_min, r_max)`
    :return: `(num_supp_levels, data_max_resource)`
    """
    num_rungs = len(rung_levels)
    assert num_rungs >= 2, "There must be at least 2 rung levels"
    num_supp_levels = num_rungs
    data_max_resource = None
    while num_supp_levels > 1:
        max_resource = rung_levels[num_supp_levels - 1]
        data_max_resource = _extract_data_at_resource(
            data=data, resource=max_resource, resource_attr_range=resource_attr_range
        )
        if data_max_resource["features"].shape[0] >= 6:
            break
        num_supp_levels -= 1
    if num_supp_levels == 1:
        data_max_resource = None
    return num_supp_levels, data_max_resource


def _extract_data_at_resource(
    data: dict, resource: int, resource_attr_range: Tuple[int, int]
) -> dict:
    features_ext = data["features"]
    targets = data["targets"]
    num_fantasies = targets.shape[1] if targets.ndim == 2 else 1
    features, resources = decode_extended_features(
        features_ext=features_ext, resource_attr_range=resource_attr_range
    )
    ind = resources == resource
    features_max = features[ind]
    targets_max = targets[ind].reshape((-1, num_fantasies))
    if num_fantasies > 1:
        # Remove pending evaluations at highest level (they are ignored). We
        # detect observed cases by all target values being the same.
        ind = np.array([x == np.full(num_fantasies, x[0]) for x in targets_max])
        features_max = features_max[ind]
        targets_max = targets_max[ind, 0]
    return {"features": features_max, "targets": targets_max.reshape((-1,))}


def _losses_for_maximum_rung_by_cross_validation(
    poster_state_for_fold: Callable[
        [np.ndarray, np.ndarray], PosteriorStateWithSampleJoint
    ],
    data_max_resource: dict,
    num_samples: int,
    random_state: Optional[RandomState],
) -> np.ndarray:
    """
    Estimates loss samples at highest rung by K-fold cross-validation, where
    `K <= 5` is chosen such that each fold has at least 2 points (since
    `len(data_max_resource) >= 6`, we have `K >= 3`).

    `poster_state_for_fold` maps training data `(features, targets)` to
    posterior state.

    For simplicity, we ignore pending evaluations here. They would affect the
    result only if they are at the highest level.

    Note that for a joint (multi-task) GP model, the per-fold models use
    restrictions of mean and covariance function learned on all data, but
    the posteriors are conditioned on the max resource data only.
    """
    features = data_max_resource["features"]
    targets = data_max_resource["targets"]
    num_data = features.shape[0]
    # K <= 5, and each fold has at least two datapoints
    num_folds = min(num_data // 2, 5)
    low_val = num_data // num_folds
    fold_sizes = np.full(num_folds, low_val)
    incr_ind = num_folds - num_data + low_val * num_folds
    fold_sizes[incr_ind:] += 1
    # Loop over folds
    result = np.zeros(num_samples)
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        train_data = {
            "features": np.vstack((features[:start], features[end:])),
            "targets": np.concatenate((targets[:start], targets[end:])),
        }
        test_data = {
            "features": features[start:end],
            "targets": targets[start:end],
        }
        start = end
        # Note: If there are pending evaluations at the highest level, they
        # are not taken into account here (no fantasizing).
        poster_state_fold = poster_state_for_fold(
            train_data["features"], train_data["targets"]
        )
        result += _losses_for_rung(
            poster_state=poster_state_fold,
            data_max_resource=test_data,
            num_samples=num_samples,
            random_state=random_state,
        )
    result *= 1 / num_folds
    return result


def _losses_for_rung(
    poster_state: PosteriorStateWithSampleJoint,
    data_max_resource: dict,
    num_samples: int,
    random_state: Optional[RandomState],
) -> np.ndarray:
    joint_sample = poster_state.sample_joint(
        test_features=data_max_resource["features"],
        num_samples=num_samples,
        random_state=random_state,
    )
    targets = data_max_resource["targets"]
    num_data = joint_sample.shape[0]
    result = np.zeros(joint_sample.shape[1:])
    for j, k in ((j, k) for j in range(num_data - 1) for k in range(j + 1, num_data)):
        yj_lt_yk = targets[j] < targets[k]
        fj_lt_fk = joint_sample[j] < joint_sample[k]
        result += np.logical_xor(fj_lt_fk, yj_lt_yk)
    result *= 2 / (num_data * (num_data - 1))
    if poster_state.num_fantasies > 1:
        assert result.ndim == 2 and result.shape == (
            poster_state.num_fantasies,
            num_samples,
        ), result.shape
        result = np.mean(result, axis=0)
    return result.reshape((-1,))
