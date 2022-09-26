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
from typing import Callable, List, Tuple, Optional

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model import (
    GaussianProcessOptimizeModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood import (
    IndependentGPPerResourceMarginalLikelihood,
)

logger = logging.getLogger(__name__)


class IndependentGPPerResourceModel(GaussianProcessOptimizeModel):
    """
    GP multi-fidelity model over f(x, r), where for each r, f(x, r) is
    represented by an independent GP. The different processes share the same
    kernel, but have their own mean functions mu_r and covariance scales c_r.

    The likelihood object is not created at construction, but only with
    `create_likelihood`. This is because we need to know the rung levels of
    the Hyperband scheduler.

    :param kernel: Kernel function without covariance scale, shared by models
        for all resources r
    :param mean_factory: Factory function for mean functions mu_r(x)
    :param resource_attr_range: (r_min, r_max)
    :param separate_noise_variances: Separate noise variance for each r?
        Otherwise, noise variance is shared
    :param initial_noise_variance: Initial value for noise variance parameter
    :param initial_covariance_scale: Initial value for covariance scale
        parameters c_r
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean_factory: Callable[[int], MeanFunction],
        resource_attr_range: Tuple[int, int],
        separate_noise_variances: bool = False,
        initial_noise_variance: Optional[float] = None,
        initial_covariance_scale: Optional[float] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        super().__init__(
            optimization_config=optimization_config,
            random_seed=random_seed,
            fit_reset_params=fit_reset_params,
        )
        self._kernel = kernel
        self._mean_factory = mean_factory
        self._resource_attr_range = resource_attr_range
        self._likelihood_kwargs = {
            "separate_noise_variances": separate_noise_variances,
            "initial_noise_variance": initial_noise_variance,
            "initial_covariance_scale": initial_covariance_scale,
        }
        self._likelihood = None  # Delayed creation

    def create_likelihood(self, rung_levels: List[int]):
        """
        Delayed creation of likelihood, needs to know rung levels of Hyperband
        scheduler.

        Note: last entry of `rung_levels` must be `max_t`, even if this is not
        a rung level in Hyperband.

        :param rung_levels: Rung levels
        """
        mean = {resource: self._mean_factory(resource) for resource in rung_levels}
        self._likelihood = IndependentGPPerResourceMarginalLikelihood(
            kernel=self._kernel,
            mean=mean,
            resource_attr_range=self._resource_attr_range,
            **self._likelihood_kwargs,
        )
        self.reset_params()

    @property
    def likelihood(self) -> MarginalLikelihood:
        assert (
            self._likelihood is not None
        ), "Call create_likelihood (passing rung levels) in order to complete creation"
        return self._likelihood
