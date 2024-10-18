from dataclasses import dataclass
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
    QuantileRegressorPredictions,
)


@dataclass
class ConformalQuantileCorrection:
    alpha: float
    sign: float = None
    correction: float = None

    def __post_init__(self):
        if self.alpha == 0.5:
            self.sign = 0.0
        elif self.alpha < 0.5:
            self.sign = 1.0
        elif self.alpha > 0.5:
            self.sign = -1.0
        else:
            raise RuntimeError(
                "Incorrect alpha provided to ConformalizedGradientBoostingQuantileRegressor"
            )


class ConformalizedGradientBoostingQuantileRegressor(GradientBoostingQuantileRegressor):
    conformal_correction: Dict[float, ConformalQuantileCorrection] = None

    def __init__(
        self,
        quantiles: Union[int, List[float]] = 9,
        valid_fraction: float = 0.10,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(quantiles, verbose, **kwargs)
        self.valid_fraction = valid_fraction
        self.conformal_correction = {
            alpha: ConformalQuantileCorrection(alpha) for alpha in self.quantiles
        }

    def fit(self, df_features: np.ndarray, y: np.array, **kwargs):
        x_training, x_validation, y_training, y_validation = train_test_split(
            df_features, y, test_size=self.valid_fraction
        )
        for quantile in tqdm(
            self.quantile_regressors,
            desc="Training Quantile Regression",
            disable=not self.verbose,
        ):
            self.quantile_regressors[quantile].fit(x_training, np.ravel(y_training))

        for alpha, cq in self.conformal_correction.items():
            residuals = cq.sign * (
                self.quantile_regressors[alpha].predict(x_validation).ravel()
                - y_validation.ravel()
            )

            if alpha < 0.5:
                target_quantile = 1 - alpha
            else:
                target_quantile = alpha

            cq.correction = np.quantile(residuals, q=target_quantile)

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        quantile_res = {
            quantile: regressor.predict(df_test)
            - self.conformal_correction[quantile].correction
            for quantile, regressor in self.quantile_regressors.items()
        }
        return QuantileRegressorPredictions.from_quantile_results(
            quantiles=self.quantiles,
            results=quantile_res,
        )
