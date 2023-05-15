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
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.predictor import (
    SklearnPredictor,
)


class SklearnEstimator:
    """
    Base class for the sklearn Estimators
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, update_params: bool
    ) -> SklearnPredictor:
        """
        Implements :meth:`fit_from_state`, given transformed data.

        :param X: Training data in ndarray of shape (n_samples, n_features)
        :param y: Target values in ndarray of shape (n_samples,)
        :param update_params: Should model (hyper)parameters be updated?
        :return: Predictor, wrapping the posterior state
        """
        raise NotImplementedError()

    @property
    def normalize_targets(self) -> bool:
        """
        :return: Should targets in ``state`` be normalized before calling
            :meth:`fit`?
        """
        return False
