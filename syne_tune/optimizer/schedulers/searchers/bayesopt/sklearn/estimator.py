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
from typing import Dict, Any
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SKLearnPredictor,
)


class SKLearnEstimator:
    """
    Base class scikit-learn based estimators, giving rise to surrogate models
    for Bayesian optimization.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, update_params: bool
    ) -> SKLearnPredictor:
        """
        Implements :meth:`fit_from_state`, given transformed data. Here,
        ``y`` is normalized (zero mean, unit variance) iff
        ``normalize_targets == True``.

        :param X: Feature matrix, shape ``(n_samples, n_features)``
        :param y: Target values, shape ``(n_samples,)``
        :param update_params: Should model (hyper)parameters be updated?
            Ignored if estimator has no hyperparameters
        :return: Predictor, wrapping the posterior state
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """
        :return: Current model hyperparameters
        """
        return dict()  # Default (estimator has no hyperparameters)

    def set_params(self, param_dict: Dict[str, Any]):
        """
        :param param_dict: New model hyperparameters
        """
        pass  # Default (estimator has no hyperparameters)

    @property
    def normalize_targets(self) -> bool:
        """
        :return: Should targets in ``state`` be normalized before calling
            :meth:`fit`?
        """
        return False
