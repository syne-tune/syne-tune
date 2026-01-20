import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from tabpfn import TabPFNRegressor

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


@dataclass
class QuantileRegressorPredictions:
    quantiles: list[float]
    results_stacked: np.ndarray

    def results(self, quantile: float) -> np.ndarray:
        assert (
            quantile in self.quantiles
        ), f"Quantile {quantile} not found in results <{self.quantiles}>"
        if type(self.quantiles) is np.ndarray:
            index = np.where(self.quantiles == quantile)[0][0]
        else:
            index = self.quantiles.index(quantile)
        return self.results_stacked[:, index]

    @classmethod
    def from_quantile_results(
        cls, quantiles: list[float], results: dict[float, np.ndarray]
    ):
        listres = [results[item] for item in quantiles]
        results_stacked = np.stack(listres, axis=1)
        return cls(quantiles=quantiles, results_stacked=results_stacked)

    @property
    def nquantiles(self) -> int:
        return len(self.quantiles)

    def mean(self) -> np.ndarray:
        for i, quantile in enumerate(self.quantiles):
            if quantile == 0.5:
                return self.results_stacked[:, i]
        return self.results_stacked.mean(axis=-1)


class QuantileRegressor:
    quantiles: list[float]

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        raise NotImplementedError()


class GradientBoostingQuantileRegressor(QuantileRegressor, GradientBoostingRegressor):
    def __init__(
        self,
        quantiles: int | list[float] = 5,
        verbose: bool = False,
        valid_fraction: float = 0.0,
        **kwargs,
    ):
        super(GradientBoostingQuantileRegressor).__init__()
        if type(quantiles) is int:
            # Compute quantiles avoiding 0-th
            quantiles = np.linspace(0, 1.0, num=quantiles + 1, endpoint=False)[1:]
            quantiles = np.around(quantiles, decimals=1 + int(np.log(len(quantiles))))
        self.quantiles = quantiles
        self.verbose = verbose
        self.valid_fraction = valid_fraction

        self.quantile_regressors = {
            quantile: GradientBoostingRegressor(
                loss="quantile", alpha=quantile, **kwargs
            )
            for quantile in quantiles
        }

    def fit(self, df_features: np.ndarray, y: np.array, **kwargs):
        if self.valid_fraction > 0.0:
            x_training, x_validation, y_training, y_validation = train_test_split(
                df_features, np.ravel(y), test_size=self.valid_fraction
            )
        else:
            x_training = df_features
            y_training = np.ravel(y)

        for quantile in tqdm(
            self.quantile_regressors,
            desc="Training Quantile Regression",
            disable=not self.verbose,
        ):
            self.quantile_regressors[quantile].fit(x_training, y_training)

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        quantile_res = {
            quantile: regressor.predict(df_test)
            for quantile, regressor in self.quantile_regressors.items()
        }
        return QuantileRegressorPredictions.from_quantile_results(
            quantiles=self.quantiles,
            results=quantile_res,
        )


class TabPFNQuantileRegressor(QuantileRegressor):
    """
    Quantile regressor using TabPFN 2.5.

    TabPFN is a foundation model for tabular data that provides native support
    for quantile predictions without requiring separate models per quantile.

    Requirements:
        pip install tabpfn

    Note: TabPFN models are gated on HuggingFace. You need to:
        1. Accept terms at https://huggingface.co/Prior-Labs/tabpfn_2_5
        2. Authenticate via `hf auth login` or set HF_TOKEN environment variable
    """

    def __init__(
        self,
        quantiles: int | list[float] = 5,
        verbose: bool = False,
        valid_fraction: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize TabPFN quantile regressor.

        :param quantiles: Number of quantiles (int) or list of quantile values.
            If int, quantiles are evenly spaced in (0, 1).
        :param verbose: Whether to print progress information.
        :param valid_fraction: Fraction of data to use for validation (unused,
            kept for API compatibility with GradientBoostingQuantileRegressor).
        :param kwargs: Additional arguments passed to TabPFNRegressor.
        """
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is not installed. Please install it with: pip install tabpfn"
            )

        if isinstance(quantiles, int):
            # Compute quantiles avoiding 0-th (same logic as GradientBoostingQuantileRegressor)
            quantiles = np.linspace(0, 1.0, num=quantiles + 1, endpoint=False)[1:]
            quantiles = np.around(quantiles, decimals=1 + int(np.log(len(quantiles))))

        self.quantiles = list(quantiles)
        self.verbose = verbose
        self.valid_fraction = valid_fraction
        self._tabpfn_kwargs = kwargs
        self._regressor: TabPFNRegressor | None = None

    def fit(self, df_features: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """
        Fit the TabPFN regressor.

        :param df_features: Training features.
        :param y: Training targets.
        :param kwargs: Additional arguments (unused, for API compatibility).
        """
        # if "HF_HOME" in os.environ:
        #     model_path = Path(os.getenv("HF_HOME")) / "tabpfn-2.5"
        # else:
        #     model_path = Path("~/.cache").expanduser() / "tabpfn-2.5"
        # model_path.mkdir(exist_ok=True, parents=True)
        self._regressor = TabPFNRegressor(**self._tabpfn_kwargs)  #, model_path=str(model_path / "tabpfn-v2.5-regressor-v2.5_default.ckpt"))
        y_train = np.ravel(y)

        if self.verbose:
            print("Fitting TabPFN regressor...")

        self._regressor.fit(df_features, y_train)

        if self.verbose:
            print("TabPFN fitting complete.")

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        """
        Predict quantiles for test data.

        :param df_test: Test features.
        :return: QuantileRegressorPredictions containing predictions for all quantiles.
        """
        if self._regressor is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # TabPFN returns a list of arrays, one per quantile
        quantile_preds_raw = self._regressor.predict(
            df_test,
            output_type="quantiles",
            quantiles=self.quantiles,
        )
        # Stack to get shape (n_samples, n_quantiles)
        quantile_preds = np.column_stack(quantile_preds_raw)

        return QuantileRegressorPredictions(
            quantiles=self.quantiles,
            results_stacked=quantile_preds,
        )
