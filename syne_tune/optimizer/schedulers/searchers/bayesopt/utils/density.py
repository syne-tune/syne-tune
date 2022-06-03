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
from math import erfc
import numpy as np
from scipy.special import erfc


def get_quantiles(acquisition_par, fmin, m, s):
    """
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    """
    if isinstance(s, np.ndarray):
        s[s < 1e-10] = 1e-10
    elif s < 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par) / s

    phi = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    # vectorized version of erfc to not depend on scipy
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)
