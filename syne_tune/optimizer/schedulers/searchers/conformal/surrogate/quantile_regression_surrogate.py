from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.surrogate_model import (
    SurrogateModel,
)
from syne_tune.blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    QuantileRegressorPredictions,
    GradientBoostingQuantileRegressor,
    TabPFNQuantileRegressor,
    TABPFN_AVAILABLE,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)

from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    subsample,
)


class QuantileRegressionSurrogateModel(SurrogateModel):
    def __init__(
        self,
        config_space: dict,
        mode: str,
        random_state: np.random.RandomState | None = None,
        max_fit_samples: int | None = None,
        quantiles: int = 5,
        valid_fraction: float = 0.0,
        min_samples_to_conformalize: int = None,
        model_type: Literal["gradient_boosting", "tabpfn"] = "gradient_boosting",
        **kwargs,
    ):
        """
        :param config_space: Configuration space for hyperparameters.
        :param mode: Optimization mode, either "min" or "max".
        :param random_state: Random state for reproducibility.
        :param max_fit_samples: Maximum number of samples to use for fitting.
        :param quantiles: Number of quantiles or list of quantile values.
        :param valid_fraction: Fraction of data to use for validation.
        :param min_samples_to_conformalize: If not None, conformalize once this
            number of samples are available (only for gradient_boosting model_type).
        :param model_type: Type of quantile regression model to use:
            - "gradient_boosting": Uses GradientBoostingQuantileRegressor (default).
              Trains separate models for each quantile using sklearn's
              GradientBoostingRegressor with quantile loss.
            - "tabpfn": Uses TabPFNQuantileRegressor with TabPFN 2.5.
              A foundation model for tabular data with native quantile support.
              Requires `pip install tabpfn` and HuggingFace authentication.
        :param kwargs: Additional arguments passed to the quantile regressor.
        """
        super(QuantileRegressionSurrogateModel, self).__init__(
            config_space=config_space,
            mode=mode,
            random_state=random_state,
            max_fit_samples=max_fit_samples,
        )

        if model_type == "tabpfn":
            if not TABPFN_AVAILABLE:
                raise ImportError(
                    "TabPFN is not installed. Please install it with: pip install tabpfn"
                )
            if min_samples_to_conformalize is not None:
                raise ValueError(
                    "min_samples_to_conformalize is not supported with TabPFN model_type. "
                    "Use model_type='gradient_boosting' for conformalized quantile regression."
                )
            quantile_regressor = TabPFNQuantileRegressor(
                quantiles=quantiles, valid_fraction=valid_fraction, **kwargs
            )
        elif model_type == "gradient_boosting":
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
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Supported types: 'gradient_boosting', 'tabpfn'"
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
