from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class QuantileRegressorPredictions:
    quantiles: List[float]
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
        cls, quantiles: List[float], results: Dict[float, np.ndarray]
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
    quantiles: List[float]

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        raise NotImplementedError()


class GradientBoostingQuantileRegressor(QuantileRegressor, GradientBoostingRegressor):
    def __init__(
        self,
        quantiles: Union[int, List[float]] = 5,
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
