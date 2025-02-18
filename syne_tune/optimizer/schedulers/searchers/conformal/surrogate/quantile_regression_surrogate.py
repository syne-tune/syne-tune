from functools import partial
from typing import Dict, Optional

import numpy as np
import pandas as pd
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.surrogate_model import (
    SurrogateModel,
)
from syne_tune.blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)

from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.legacy_quantile_based_searcher import (
    subsample,
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
        # only consider non-constant parts of the config space
        hp_cols = [k for k, v in self.config_space.items() if hasattr(v, "sample")]
        self.model_pipeline = BlackboxSurrogate.make_model_pipeline(
            configuration_space={
                k: v for k, v in self.config_space.items() if k in hp_cols
            },
            fidelity_space={},
            model=self.quantile_regressor,
        )
        X_train, y_train = subsample(
            df_features.loc[:, hp_cols],
            y,
            max_samples=self.max_fit_samples,
            random_state=self.random_state,
        )
        self.model_pipeline.fit(X_train, y_train)

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
