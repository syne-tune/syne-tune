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
# This file contains various constants required for the definition of the model
# or to set up the optimization

import autograd.numpy as anp
from dataclasses import dataclass

DEFAULT_ENCODING = "logarithm"  # the other choices is positive

NUMERICAL_JITTER = 1e-9

INITIAL_NOISE_VARIANCE = 1e-3
INITIAL_MEAN_VALUE = 0.0
INITIAL_COVARIANCE_SCALE = 1.0
INITIAL_INVERSE_BANDWIDTHS = 1.0
INITIAL_WARPING = 1.0

INVERSE_BANDWIDTHS_LOWER_BOUND = 1e-4
INVERSE_BANDWIDTHS_UPPER_BOUND = 100

COVARIANCE_SCALE_LOWER_BOUND = 1e-3
COVARIANCE_SCALE_UPPER_BOUND = 1e3

NOISE_VARIANCE_LOWER_BOUND = 1e-9
NOISE_VARIANCE_UPPER_BOUND = 1e6

WARPING_LOWER_BOUND = 0.25
WARPING_UPPER_BOUND = 4.0

MIN_POSTERIOR_VARIANCE = 1e-12

MIN_CHOLESKY_DIAGONAL_VALUE = 1e-10

DATA_TYPE = anp.float64


@dataclass
class OptimizationConfig:
    lbfgs_tol: float
    lbfgs_maxiter: int
    verbose: bool
    n_starts: int


@dataclass
class MCMCConfig:
    """
    `n_samples` is the total number of samples drawn. The first `n_burnin` of
    these are dropped (burn-in), and every `n_thinning` of the rest is
    returned. This means we return
    `(n_samples - n_burnin) // n_thinning` samples.
    """

    n_samples: int
    n_burnin: int
    n_thinning: int


DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig(
    lbfgs_tol=1e-6, lbfgs_maxiter=500, verbose=False, n_starts=5
)

DEFAULT_MCMC_CONFIG = MCMCConfig(n_samples=300, n_burnin=250, n_thinning=5)
