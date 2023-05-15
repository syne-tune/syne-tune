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
from typing import Optional, List, Dict

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BasePredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SklearnPredictor,
)


class SklearnPredictorWrapper(BasePredictor):
    """
    Wrapper class for the sklearn estimators to be used with ContributedEstimatorWrapper
    """

    def __init__(
        self,
        contributed_predictor: SklearnPredictor,
        state: TuningJobState,
        active_metric: Optional[str] = None,
    ):
        super().__init__(state, active_metric)
        self.contributed_predictor = contributed_predictor

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Returns signals which are statistics of the predictive distribution at
        input points ``inputs``. By default:

        * "mean": Predictive means.
        * "std": Predictive stddevs, shape ``(n,)``

        This function relies on the assigned ContributedPredictor to execute the predictions

        :param inputs: Input points, shape ``(n, d)``
        :return: List of ``dict`` with keys :meth:`keys_predict`, of length 1
        """

        mean, std = self.contributed_predictor.predict(inputs)
        outputs = {"mean": mean}
        if std is not None:
            outputs["std"] = std
        return [outputs]

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
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
        :return: Gradient :math:`\nabla f(x)` (one-length list)
        """
        return [
            self.contributed_predictor.backward_gradient(
                input=input, head_gradients=head_gradients
            )
        ]
