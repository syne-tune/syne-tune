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
from functools import partial
from typing import Dict, Optional

import numpy as np
import pandas as pd
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.surrogate_model import (
    SurrogateModel,
)

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    fit_model,
)


class QuantileRegressionSurrogateModel(SurrogateModel):
    def __init__(
        self,
        config_space: Dict,
        mode: str,
        random_state: Optional[np.random.RandomState] = None,
        max_fit_samples: Optional[int] = None,
        quantiles: int = 5,
        valid_fraction: float = 0.0,
        min_samples_to_conformalize: int = None,
        **kwargs,
    ):
        """
        :param min_samples_to_conformalize: if value is not None, conformalize once this number of samples are available
        """
        super(QuantileRegressionSurrogateModel, self).__init__(
            config_space=config_space,
            mode=mode,
            random_state=random_state,
            max_fit_samples=max_fit_samples,
        )
        if min_samples_to_conformalize is not None:
            quantile_regressor_cls = partial(
                SymmetricConformalizedGradientBoostingQuantileRegressor,
                min_samples_to_conformalize=min_samples_to_conformalize,
            )
        else:
            quantile_regressor_cls = GradientBoostingQuantileRegressor
        quantile_regressor = quantile_regressor_cls(
            quantiles=quantiles, valid_fraction=valid_fraction, **kwargs
        )
        self.quantile_regressor = quantile_regressor
        self.model_pipeline = None

    def _fit(self, df_features: pd.DataFrame, y: np.array):
        model_pipeline, sigma_train, _ = fit_model(
            X=df_features,
            y=y,
            max_fit_samples=self.max_fit_samples,
            config_space=self.config_space,
            random_state=self.random_state,
            model=self.quantile_regressor,
            max_val_samples=None,
            do_eval_model=False,
        )
        self.model_pipeline = model_pipeline

    def _sample_best(self) -> int:
        residual_samples = self._surrogate_pred()
        if self.mode == "max":
            residual_samples *= -1
        return np.argmin(residual_samples)

    def _get_sampler(self, df_features: pd.DataFrame) -> np.array:
        quantileResults = self.predict(df_features)

        def sampler():
            sampled_indexes = np.random.randint(
                low=0,
                high=quantileResults.nquantiles,
                size=len(quantileResults.results_stacked),
            )
            columns = np.arange(len(quantileResults.results_stacked))
            return quantileResults.results_stacked[columns, sampled_indexes]

        return sampler

    def predict(self, df_features: pd.DataFrame) -> QuantileRegressorPredictions:
        """
        This will need quantiles, median, mean
        """
        quantiles = self.model_pipeline.predict(df_features)
        return quantiles
