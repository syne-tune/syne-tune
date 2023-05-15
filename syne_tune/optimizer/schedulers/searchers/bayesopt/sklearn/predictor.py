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
from typing import Tuple, List, Dict

import numpy as np


class SklearnPredictor:
    """
    Base class for the sklearn predictors
    """

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns signals which are statistics of the predictive distribution at
        input points ``inputs``.


        :param inputs: Input points, shape ``(n, d)``
        :return: Tuple with the following entries:
            * "mean": Predictive means in shape of ``(n,)``
            * "std": Predictive stddevs, shape ``(n,)``
        """

        raise NotImplementedError()

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Computes the gradient :math:`\nabla f(x)` for an acquisition
        function :math:`f(x)`, where :math:`x` is a single input point. This
        is using reverse mode differentiation, the head gradients are passed
        by the acquisition function. The head gradients are
        :math:`\partial_k f`, where :math:`k` runs over the statistics
        returned by :meth:`predict` for the single input point :math:`x`.
        The shape of head gradients is the same as the shape of the
        statistics.

        :param input: Single input point :math:`x`, shape ``(d,)``
        :param head_gradients: See above
        :return: Gradient :math:`\nabla f(x)`
        """
        raise NotImplementedError()
