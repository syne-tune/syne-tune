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
import autograd.numpy as anp
from autograd.builtins import isinstance
import numbers
from scipy.special import gammaln

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    MIN_POSTERIOR_VARIANCE,
)

__all__ = ["Distribution", "Gamma", "Uniform", "Normal", "LogNormal", "Horseshoe"]


class Distribution:
    def negative_log_density(self, x):
        """
        Negative log density. lower and upper limits are ignored.
        If x is not a scalar, the distribution is i.i.d. over all
        entries.
        """
        raise NotImplementedError


class Gamma(Distribution):
    """
    Gamma(mean, alpha):

        p(x) = C(alpha, beta) x^{alpha - 1} exp( -beta x), beta = alpha / mean,
        C(alpha, beta) = beta^alpha / Gamma(alpha)
    """

    def __init__(self, mean, alpha):
        self._assert_positive_number(mean, "mean")
        self._assert_positive_number(alpha, "alpha")
        self.mean = anp.maximum(mean, MIN_POSTERIOR_VARIANCE)
        self.alpha = anp.maximum(alpha, MIN_POSTERIOR_VARIANCE)
        self.beta = self.alpha / self.mean
        self.log_const = gammaln(self.alpha) - self.alpha * anp.log(self.beta)
        self.__call__ = self.negative_log_density

    @staticmethod
    def _assert_positive_number(x, name):
        assert (
            isinstance(x, numbers.Real) and x > 0.0
        ), "{} = {}, must be positive number".format(name, x)

    def negative_log_density(self, x):
        x_safe = anp.maximum(x, MIN_POSTERIOR_VARIANCE)
        return anp.sum(
            (1.0 - self.alpha) * anp.log(x_safe) + self.beta * x_safe + self.log_const
        )

    def __call__(self, x):
        return self.negative_log_density(x)


class Uniform(Distribution):
    def __init__(self, lower: float, upper: float):
        self.log_const = anp.log(upper - lower)
        self.__call__ = self.negative_log_density

    def negative_log_density(self, x):
        return x.size * self.log_const

    def __call__(self, x):
        return self.negative_log_density(x)


class Normal(Distribution):
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        self.__call__ = self.negative_log_density

    def negative_log_density(self, x):
        return anp.sum(anp.square(x - self.mean)) * (0.5 / anp.square(self.sigma))

    def __call__(self, x):
        return self.negative_log_density(x)


class LogNormal(Distribution):
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        self.__call__ = self.negative_log_density

    def negative_log_density(self, x):
        x_safe = anp.maximum(x, MIN_POSTERIOR_VARIANCE)
        return anp.sum(
            anp.log(x_safe * self.sigma)
            + anp.square(anp.log(x_safe) - self.mean) * (0.5 / anp.square(self.sigma))
        )

    def __call__(self, x):
        return self.negative_log_density(x)


class Horseshoe(Distribution):
    def __init__(self, s: float):
        assert s > 0.0
        self.s = max(s, MIN_POSTERIOR_VARIANCE)
        self.__call__ = self.negative_log_density

    def negative_log_density(self, x):
        arg = anp.maximum(3.0 * anp.square(self.s / x), MIN_POSTERIOR_VARIANCE)
        return -anp.sum(anp.log(anp.log1p(arg)))

    def __call__(self, x):
        return self.negative_log_density(x)
