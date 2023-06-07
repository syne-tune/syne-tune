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
from typing import Optional
import logging

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
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    ScalarTargetTransform,
)

logger = logging.getLogger(__name__)


class GaussianProcessRegression(GaussianProcessOptimizeModel):
    """
    Gaussian Process Regression

    Takes as input a mean function (which depends on X only) and a kernel
    function.

    :param kernel: Kernel function
    :param mean: Mean function which depends on the input X only (by default,
        a scalar fitted while optimizing the likelihood)
    :param target_transform: Invertible transform of target values y to
        latent values z, which are then modelled as Gaussian. Defaults to
        the identity
    :param initial_noise_variance: Initial value for noise variance parameter
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: Optional[MeanFunction] = None,
        target_transform: Optional[ScalarTargetTransform] = None,
        initial_noise_variance: Optional[float] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        super().__init__(
            optimization_config=optimization_config,
            random_seed=random_seed,
            fit_reset_params=fit_reset_params,
        )
        self._likelihood = GaussianProcessMarginalLikelihood(
            kernel=kernel,
            mean=mean,
            target_transform=target_transform,
            initial_noise_variance=initial_noise_variance,
        )
        self.reset_params()

    @property
    def likelihood(self) -> MarginalLikelihood:
        return self._likelihood
