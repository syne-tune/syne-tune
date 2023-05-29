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
from typing import Tuple, Dict, Union

import numpy as np


class SKLearnPredictor:
    """
    Base class for the sklearn predictors.
    """

    @staticmethod
    def returns_std() -> bool:
        """
        :return: Does :meth:`predict` return stddevs as well? Otherwise, only
            means (i.e., point estimates) are returned
        """
        raise NotImplementedError

    def predict(
        self, X: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns signals which are statistics of the predictive distribution at
        input points ``inputs``.

        :param inputs: Input points, shape ``(n, d)``
        :return: ``(means, stds)`` if ``returns_std() == True``, or
            ``means`` otherwise, where predictive means ``means`` and
            predictive stddevs ``stds`` have shape ``(n,)``
        """
        raise NotImplementedError

    def backward_gradient(
        self, input: np.ndarray, head_gradients: Dict[str, np.ndarray]
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
        raise NotImplementedError
