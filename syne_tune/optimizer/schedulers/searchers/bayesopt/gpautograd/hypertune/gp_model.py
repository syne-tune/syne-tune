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
from typing import Callable, Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.likelihood import (
    HyperTuneIndependentGPMarginalLikelihood,
    HyperTuneJointGPMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.utils import (
    number_supported_levels_and_data_highest_level,
    hypertune_ranking_losses,
    GaussProcPosteriorStateAndRungLevels,
    PerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model import (
    IndependentGPPerResourceModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression import (
    GaussianProcessRegression,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
    ScalarMeanFunction,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler


@dataclass
class HyperTuneDistributionArguments:
    num_samples: int
    num_brackets: Optional[int] = None

    def __post_init__(self):
        assert self.num_brackets is None or self.num_brackets >= 1
        assert self.num_samples >= 1


class HyperTuneModelMixin:
    def __init__(self, hypertune_distribution_args: HyperTuneDistributionArguments):
        self.hypertune_distribution_args = hypertune_distribution_args
        self._bracket_distribution = None
        # Tuple (num_supp_levels, data_size) for current distribution. If
        # this signature is different in `fit`, the distribution is recomputed
        self._hypertune_distribution_signature = None

    def hypertune_bracket_distribution(self) -> Optional[np.ndarray]:
        """
        Distribution [w_k] of support size `num_supp_brackets`, where
        `num_supp_brackets <= args.num_brackets` (the latter is maximum if
        not given) is maximum such that the first `num_supp_brackets`
        brackets have >= 6 labeled datapoints each.

        If `num_supp_brackets < args.num_brackets`, the distribution must be
        extended to full size before being used to sample the next bracket.
        """
        return self._bracket_distribution

    def hypertune_ensemble_distribution(self) -> Optional[Dict[int, float]]:
        """
        Distribution [theta_r] which is used to create an ensemble predictive
        distribution fed into the acquisition function.
        The ensemble distribution runs over all sufficiently supported rung
        levels, independent of the number of brackets.
        """
        raise NotImplementedError

    def fit_distributions(
        self,
        poster_state: PerResourcePosteriorState,
        data: dict,
        resource_attr_range: Tuple[int, int],
        random_state: np.random.RandomState,
    ) -> Optional[Dict[int, float]]:
        ensemble_distribution = None
        args = self.hypertune_distribution_args
        (
            num_supp_levels,
            data_resource,
        ) = number_supported_levels_and_data_highest_level(
            rung_levels=poster_state.rung_levels,
            data=data,
            resource_attr_range=resource_attr_range,
        )
        if num_supp_levels > 1:
            num_data = data_resource["features"].shape[0]
            curr_sig = self._hypertune_distribution_signature
            new_sig = (num_supp_levels, num_data)
            if curr_sig is None or new_sig != curr_sig:
                # Data at highest level has changed
                self._hypertune_distribution_signature = new_sig
                ranking_losses = hypertune_ranking_losses(
                    poster_state=poster_state,
                    data=data,
                    num_samples=args.num_samples,
                    resource_attr_range=resource_attr_range,
                    random_state=random_state,
                )
                min_losses = np.min(ranking_losses, axis=0, keepdims=True)
                theta = np.sum(ranking_losses == min_losses, axis=1).reshape((-1,))
                theta = theta / np.sum(theta)
                # We sparsify the ensemble distribution
                rung_levels = poster_state.rung_levels[: theta.size]
                ensemble_distribution = {
                    resource: theta_val
                    for resource, theta_val in zip(rung_levels, theta)
                    if theta_val > 0.01
                }
                self._bracket_distribution = theta * np.array(
                    [1 / r for r in rung_levels]
                )
                if args.num_brackets < theta.size:
                    self._bracket_distribution = self._bracket_distribution[
                        : args.num_brackets
                    ]
                norm_const = np.sum(self._bracket_distribution)
                if norm_const > 1e-14:
                    self._bracket_distribution /= norm_const
                else:
                    self._bracket_distribution[:] = 0.0
                    self._bracket_distribution[0] = 1.0

        return ensemble_distribution


class HyperTuneIndependentGPModel(IndependentGPPerResourceModel, HyperTuneModelMixin):
    """
    Variant of :class:`IndependentGPPerResourceModel` which implements additional
    features of the Hyper-Tune algorithm, see

        Yang Li et al
        Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale
        VLDB 2022

    Our implementation differs from the Hyper-Tune paper in a number of ways.
    Most importantly, their method requires a sufficient number of observed
    points at the starting rung of the highest bracket. In contrast, we
    estimate ranking loss values already when the starting rung of the 2nd
    bracket is sufficiently occupied. This allows us to estimate the head
    of the distribution only (over all brackets with sufficiently occupied
    starting rungs), and we use the default distribution over the remaining
    tail. Eventually, we do the same as Hyper-Tune, but we move away from the
    default distribution earlier on.

    :param hypertune_distribution_args: Parameters for Hyper-Tune
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean_factory: Callable[[int], MeanFunction],
        resource_attr_range: Tuple[int, int],
        hypertune_distribution_args: HyperTuneDistributionArguments,
        separate_noise_variances: bool = False,
        initial_noise_variance: Optional[float] = None,
        initial_covariance_scale: Optional[float] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        IndependentGPPerResourceModel.__init__(
            self,
            kernel=kernel,
            mean_factory=mean_factory,
            resource_attr_range=resource_attr_range,
            separate_noise_variances=separate_noise_variances,
            initial_noise_variance=initial_noise_variance,
            initial_covariance_scale=initial_covariance_scale,
            optimization_config=optimization_config,
            random_seed=random_seed,
            fit_reset_params=fit_reset_params,
        )
        HyperTuneModelMixin.__init__(
            self, hypertune_distribution_args=hypertune_distribution_args
        )

    def create_likelihood(self, rung_levels: List[int]):
        """
        Delayed creation of likelihood, needs to know rung levels of Hyperband
        scheduler.

        Note: last entry of `rung_levels` must be `max_t`, even if this is not
        a rung level in Hyperband.

        :param rung_levels: Rung levels
        """
        mean = {resource: self._mean_factory(resource) for resource in rung_levels}
        # Safe bet to start with:
        ensemble_distribution = {min(rung_levels): 1.0}
        self._likelihood = HyperTuneIndependentGPMarginalLikelihood(
            kernel=self._kernel,
            mean=mean,
            resource_attr_range=self._resource_attr_range,
            ensemble_distribution=ensemble_distribution,
            **self._likelihood_kwargs,
        )
        self.reset_params()

    def hypertune_ensemble_distribution(self) -> Optional[Dict[int, float]]:
        if self._likelihood is not None:
            return self._likelihood.ensemble_distribution
        else:
            return None

    def fit(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        super().fit(data, profiler)
        poster_state: IndependentGPPerResourcePosteriorState = self.states[0]
        ensemble_distribution = self.fit_distributions(
            poster_state=poster_state,
            data=data,
            resource_attr_range=self._resource_attr_range,
            random_state=self.random_state,
        )
        if ensemble_distribution is not None:
            # Recompute posterior state (likelihood changed)
            self._likelihood.set_ensemble_distribution(ensemble_distribution)
            self._recompute_states(data)


class HyperTuneJointGPModel(GaussianProcessRegression, HyperTuneModelMixin):
    """
    Variant of :class:`GaussianProcessRegression` which implements additional
    features of the Hyper-Tune algorithm, see

        Yang Li et al
        Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale
        VLDB 2022

    See also :class:`HyperTuneIndependentGPModel`

    :param hypertune_distribution_args: Parameters for Hyper-Tune
    """

    def __init__(
        self,
        kernel: KernelFunction,
        resource_attr_range: Tuple[int, int],
        hypertune_distribution_args: HyperTuneDistributionArguments,
        mean: Optional[MeanFunction] = None,
        initial_noise_variance: Optional[float] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        if mean is None:
            mean = ScalarMeanFunction()
        GaussianProcessRegression.__init__(
            self,
            kernel=kernel,
            mean=mean,
            initial_noise_variance=initial_noise_variance,
            optimization_config=optimization_config,
            random_seed=random_seed,
            fit_reset_params=fit_reset_params,
        )
        HyperTuneModelMixin.__init__(
            self, hypertune_distribution_args=hypertune_distribution_args
        )
        self._likelihood_kwargs = dict(
            kernel=kernel,
            mean=mean,
            resource_attr_range=resource_attr_range,
            initial_noise_variance=initial_noise_variance,
        )
        self._likelihood = None
        self._rung_levels = None

    def create_likelihood(self, rung_levels: List[int]):
        """
        Delayed creation of likelihood, needs to know rung levels of Hyperband
        scheduler.

        Note: last entry of `rung_levels` must be `max_t`, even if this is not
        a rung level in Hyperband.

        :param rung_levels: Rung levels
        """
        self._rung_levels = rung_levels.copy()
        # Safe bet to start with:
        ensemble_distribution = {min(rung_levels): 1.0}
        self._likelihood = HyperTuneJointGPMarginalLikelihood(
            ensemble_distribution=ensemble_distribution,
            **self._likelihood_kwargs,
        )
        self.reset_params()

    def hypertune_ensemble_distribution(self) -> Optional[Dict[int, float]]:
        if self._likelihood is not None:
            return self._likelihood.ensemble_distribution
        else:
            return None

    def fit(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        super().fit(data, profiler)
        resource_attr_range = self._likelihood_kwargs["resource_attr_range"]
        poster_state = GaussProcPosteriorStateAndRungLevels(
            poster_state=self.states[0],
            rung_levels=self._rung_levels,
        )
        ensemble_distribution = self.fit_distributions(
            poster_state=poster_state,
            data=data,
            resource_attr_range=resource_attr_range,
            random_state=self.random_state,
        )
        if ensemble_distribution is not None:
            # Recompute posterior state (likelihood changed)
            self._likelihood.set_ensemble_distribution(ensemble_distribution)
            self._recompute_states(data)
