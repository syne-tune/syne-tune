from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
    QuantileRegressor,
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


def symmetric_conformal_corrections(
    predictions: QuantileRegressorPredictions, y: np.ndarray
) -> dict[float, float]:
    """
    Computes split conformal corrections from quantiles predicted on validation data.

    Quantiles are paired symmetrically (e.g. 0.1 with 0.9), and for each pair the
    correction is chosen so that the band widened by it covers ``1 - 2 * lower``
    of the validation targets.

    :param predictions: quantiles predicted on the validation inputs
    :param y: validation targets
    :return: correction for each pair, keyed by the lower quantile
    """
    y = np.ravel(y)
    quantiles = sorted(predictions.quantiles)
    corrections = {}
    for idx in range(len(quantiles) // 2):
        cq = SymmetricConformalQuantileCorrection(
            lower_quantile=quantiles[idx],
            upper_quantile=quantiles[len(quantiles) - 1 - idx],
        )
        residuals_lower = predictions.results(cq.lower_quantile).ravel() - y
        residuals_upper = y - predictions.results(cq.upper_quantile).ravel()
        residuals = np.maximum(residuals_lower, residuals_upper)
        # Finite-sample split conformal quantile, which guarantees the target
        # coverage. When n is too small for the guarantee, the level is capped at 1,
        # which takes the largest residual.
        n = residuals.size
        level = min(1.0, np.ceil((n + 1) * cq.target_coverage) / n)
        corrections[cq.lower_quantile] = np.quantile(
            residuals, q=level, method="higher"
        )
    return corrections


def apply_symmetric_conformal_corrections(
    predictions: QuantileRegressorPredictions, corrections: dict[float, float]
) -> QuantileRegressorPredictions:
    """
    Widens (or shrinks) quantile predictions with corrections computed by
    :func:`symmetric_conformal_corrections`. Quantiles without a correction, such
    as the median, are left unchanged.
    """
    quantiles = sorted(predictions.quantiles)
    quantile_res = {q: predictions.results(q) for q in predictions.quantiles}
    for idx in range(len(quantiles) // 2):
        lower, upper = quantiles[idx], quantiles[len(quantiles) - 1 - idx]
        correction = corrections.get(lower, 0.0)
        quantile_res[lower] = quantile_res[lower] - correction
        quantile_res[upper] = quantile_res[upper] + correction
    return QuantileRegressorPredictions.from_quantile_results(
        quantiles=predictions.quantiles,
        results=quantile_res,
    )


class SymmetricConformalizedQuantileRegressor(QuantileRegressor):
    """
    Applies split conformal calibration to any quantile regressor, for instance
    TabPFN or TabICL. Once more than ``min_samples_to_conformalize`` observations
    are available, ``valid_fraction`` of them are held out to compute corrections
    and the base regressor is fitted on the rest.
    """

    def __init__(
        self,
        base_regressor: QuantileRegressor,
        valid_fraction: float = 0.10,
        min_samples_to_conformalize: int = 32,
    ):
        assert valid_fraction > 0
        self.base_regressor = base_regressor
        self.quantiles = base_regressor.quantiles
        self.valid_fraction = valid_fraction
        self.min_samples_to_conformalize = min_samples_to_conformalize
        self.corrections = {}

    def fit(self, df_features: np.ndarray, y: np.array, **kwargs):
        nsamples = len(np.ravel(y))
        if (nsamples > self.min_samples_to_conformalize) or (
            self.min_samples_to_conformalize == 0
        ):
            x_training, x_validation, y_training, y_validation = train_test_split(
                df_features, y, test_size=self.valid_fraction
            )
            self.base_regressor.fit(x_training, y_training, **kwargs)
            self.corrections = symmetric_conformal_corrections(
                self.base_regressor.predict(x_validation), y_validation
            )
        else:
            self.base_regressor.fit(df_features, y, **kwargs)
            self.corrections = {}
        self._is_fitted = True
        return self

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        check_is_fitted(self)
        return apply_symmetric_conformal_corrections(
            self.base_regressor.predict(df_test), self.corrections
        )

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


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

        self._is_fitted = True
        return self

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

        validation_predictions = self._predict_uncorrected(x_validation)
        corrections = symmetric_conformal_corrections(
            validation_predictions, y_validation
        )
        for alpha, cq in self.conformal_correction.items():
            cq.correction = corrections[cq.lower_quantile]

    def _predict_uncorrected(self, df_test) -> QuantileRegressorPredictions:
        return QuantileRegressorPredictions.from_quantile_results(
            quantiles=self.quantiles,
            results={
                quantile: regressor.predict(df_test)
                for quantile, regressor in self.quantile_regressors.items()
            },
        )

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        check_is_fitted(self)
        return apply_symmetric_conformal_corrections(
            self._predict_uncorrected(df_test),
            {
                cq.lower_quantile: cq.correction
                for cq in self.conformal_correction.values()
            },
        )

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
