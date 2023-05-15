from typing import Tuple, List, Dict, Optional

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import BasePredictor


class ContributedPredictor:
    """
    Base class for the contributed predictors
    """

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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


class ContributedPredictorWrapper(BasePredictor):
    """
    Wrapper class for the contributed estimators to be used with ContributedEstimatorWrapper
    """

    def __init__(
            self,
            contributed_predictor: ContributedPredictor,
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
        - "std": Predictive stddevs, shape ``(n,)``

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
        return [self.contributed_predictor.backward_gradient(
            input=input,
            head_gradients=head_gradients
        )]
