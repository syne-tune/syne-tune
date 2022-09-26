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

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.likelihood import (
    GaussAdditiveMarginalLikelihood,
    LCModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model import (
    GaussianProcessOptimizeModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    ScalarMeanFunction,
    MeanFunction,
)

logger = logging.getLogger(__name__)


class GaussianProcessLearningCurveModel(GaussianProcessOptimizeModel):
    """
    Represents joint Gaussian model of learning curves over a number of
    configurations. The model has an additive form:

        f(x, r) = g(r | x) + h(x),

    where h(x) is a Gaussian process model for function values at r_max, and
    the g(r | x) are independent Gaussian models. Right now, g(r | x) can be:

    - Innovation state space model (ISSM) of a particular power-law decay
        form. For this one, g(r_max | x) = 0 for all x. Used if
        `res_model` is of type :class:`ISSModelParameters`
    - Gaussian process model with exponential decay covariance function. This
        is essentially the model from the Freeze Thaw paper, see also
        :class:`ExponentialDecayResourcesKernelFunction`. Used if
        `res_model` is of type :class:`ExponentialDecayBaseKernelFunction`

    Importantly, inference scales cubically only in the number of
    configurations, not in the number of observations.

    Details about ISSMs in general are found in

        Hyndman, R. and Koehler, A. and Ord, J. and Snyder, R.
        Forecasting with Exponential Smoothing: The State Space Approach
        Springer, 2008

    :param kernel: Kernel function k(X, X')
    :param res_model: Model for g(r | x)
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
        self,
        kernel: KernelFunction,
        res_model: LCModel,
        mean: Optional[MeanFunction] = None,
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
        if mean is None:
            mean = ScalarMeanFunction()
        self._likelihood = GaussAdditiveMarginalLikelihood(
            kernel=kernel,
            res_model=res_model,
            mean=mean,
            initial_noise_variance=initial_noise_variance,
        )
        self.reset_params()

    @property
    def likelihood(self) -> MarginalLikelihood:
        return self._likelihood
