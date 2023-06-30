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
from typing import Dict, Any, Optional, List

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BasePredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import (
    Estimator,
    transform_state_to_data,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn import (
    SKLearnPredictor,
    SKLearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
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

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        means, stddevs = self.sklearn_predictor.predict(X=inputs)
        return [{"mean": means, "std": stddevs}]

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        r"""
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


class SKLearnEstimatorWrapper(Estimator):
    """
    Wrapper class for sklearn estimators.
    """

    def __init__(
        self,
        sklearn_estimator: SKLearnEstimator,
        active_metric: str = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sklearn_estimator = sklearn_estimator
        self.target_metric = active_metric

    def get_params(self) -> Dict[str, Any]:
        return self.sklearn_estimator.get_params()

    def set_params(self, param_dict: Dict[str, Any]):
        self.sklearn_estimator.set_params(param_dict)

    def fit_from_state(self, state: TuningJobState, update_params: bool) -> Predictor:
        """
        Creates a
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
        object based on data in ``state``.

        If the model also has hyperparameters, these are learned iff
        ``update_params == True``. Otherwise, these parameters are not changed,
        but only the posterior state is computed.
        If your surrogate model is not Bayesian, or does not have hyperparameters,
        you can ignore the ``update_params`` argument.

        If ``self.state.pending_evaluations`` is not empty, we compute posterior for state without pending evals.
        This method can be overwritten for any other behaviour such as one found in
        :meth:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcEstimator.fit_from_state`.

        :param state: Current data model parameters are to be fit on, and the
            posterior state is to be computed from
        :param update_params: See above
        :return: Predictor, wrapping the posterior state
        """
        if state.pending_evaluations:
            # Remove pending evaluations
            state = TuningJobState(
                hp_ranges=state.hp_ranges,
                config_for_trial=state.config_for_trial,
                trials_evaluations=state.trials_evaluations,
                failed_trials=state.failed_trials,
            )

        data = transform_state_to_data(
            state=state,
            active_metric=self.target_metric,
            normalize_targets=self.sklearn_estimator.normalize_targets,
        )
        sklearn_predictor = self.sklearn_estimator.fit(
            X=data.features, y=data.targets, update_params=update_params
        )
        return SKLearnPredictorWrapper(sklearn_predictor=sklearn_predictor, state=state)
