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
from typing import Optional, List, Dict, Set

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BasePredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SKLearnPredictor,
)


class SKLearnPredictorWrapper(BasePredictor):
    """
    Wrapper class for sklearn predictors returned by ``fit_from_state`` of
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_estimator.SKLearnEstimatorWrapper`.
    """

    def __init__(
        self,
        sklearn_predictor: SKLearnPredictor,
        state: TuningJobState,
        active_metric: Optional[str] = None,
    ):
        super().__init__(state, active_metric)
        self.sklearn_predictor = sklearn_predictor

    def keys_predict(self) -> Set[str]:
        if self.sklearn_predictor.returns_std():
            return {"mean", "std"}
        else:
            return {"mean"}

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        prediction = self.sklearn_predictor.predict(X=inputs)
        if self.sklearn_predictor.returns_std():
            outputs = {"mean": prediction[0], "std": prediction[1]}
        else:
            outputs = {"mean": prediction}
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
        assert len(head_gradients) == 1
        return [
            self.sklearn_predictor.backward_gradient(
                input=input, head_gradients=head_gradients[0]
            )
        ]
