from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
    QuantileRegressorPredictions,
)


@dataclass
class SymmetricConformalQuantileCorrection:
    lower_quantile: float
    upper_quantile: float
    target_coverage: float = None
    correction: float = None

    def __post_init__(self):
        assert abs(1 - self.lower_quantile - self.upper_quantile) < 1e-7, (
            f"Upper and lower quantiles must be symmetric and "
            f"<{self.lower_quantile}, {self.upper_quantile}> are given"
        )
        self.target_coverage = 1 - 2 * self.lower_quantile


class SymmetricConformalizedGradientBoostingQuantileRegressor(
    GradientBoostingQuantileRegressor
):
    def __init__(
        self,
        quantiles: int = 5,
        valid_fraction: float = 0.10,
        min_samples_to_conformalize: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        assert (
            type(quantiles) is int
        ), "This class only accepts the total number of quantiles"
        assert quantiles % 2 == 1, "The number of quantiles must be odd"
        assert valid_fraction > 0
        super().__init__(quantiles, verbose, **kwargs)
        self.valid_fraction = valid_fraction
        self.min_samples_to_conformalize = min_samples_to_conformalize
        quantile_bands_locations = sorted(list(self.quantile_regressors.keys()))
        ncorrections = (quantiles - 1) // 2

        self.conformal_correction = {
            quantile_bands_locations[idx]: SymmetricConformalQuantileCorrection(
                lower_quantile=quantile_bands_locations[idx],
                upper_quantile=quantile_bands_locations[
                    len(quantile_bands_locations) - 1 - idx
                ],
            )
            for idx in range(ncorrections)
        }

    def fit(self, df_features: np.ndarray, y: np.array, **kwargs):
        nsamples = len(np.ravel(y))
        if (nsamples > self.min_samples_to_conformalize) or (
            self.min_samples_to_conformalize == 0
        ):
            self._fit_conformal(df_features, y, **kwargs)
        else:
            self._fit_nonconformal(df_features, y, **kwargs)

    def _fit_nonconformal(self, df_features: np.ndarray, y: np.array, **kwargs):
        for quantile in tqdm(
            self.quantile_regressors,
            desc="Training Quantile Regression",
            disable=not self.verbose,
        ):
            self.quantile_regressors[quantile].fit(df_features, np.ravel(np.ravel(y)))

        for alpha, cq in self.conformal_correction.items():
            cq.correction = 0

    def _fit_conformal(self, df_features: np.ndarray, y: np.array, **kwargs):
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
            residuals_lower = (
                self.quantile_regressors[cq.lower_quantile]
                .predict(x_validation)
                .ravel()
                - y_validation.ravel()
            )
            residuals_upper = (
                y_validation.ravel()
                - self.quantile_regressors[cq.upper_quantile]
                .predict(x_validation)
                .ravel()
            )
            residuals = np.maximum(residuals_lower, residuals_upper)
            cq.correction = np.quantile(
                residuals, q=cq.target_coverage / (1 + 1 / residuals.size)
            )

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        quantile_res = {}
        for alpha, cq in self.conformal_correction.items():
            lower_preds = (
                self.quantile_regressors[cq.lower_quantile].predict(df_test)
                - cq.correction
            )
            quantile_res[cq.lower_quantile] = lower_preds

            upper_preds = (
                self.quantile_regressors[cq.upper_quantile].predict(df_test)
                + cq.correction
            )
            quantile_res[cq.upper_quantile] = upper_preds

        # Predict for 0.5-th quantile since it is not corrected
        quantile_res[0.5] = self.quantile_regressors[0.5].predict(df_test)
        return QuantileRegressorPredictions.from_quantile_results(
            quantiles=self.quantiles,
            results=quantile_res,
        )
