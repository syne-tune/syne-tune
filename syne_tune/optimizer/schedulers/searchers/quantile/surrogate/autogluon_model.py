from autogluon.tabular import TabularDataset, TabularPredictor
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from syne_tune.optimizer.schedulers.searchers.quantile.surrogate.surrogate import Model
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    fit_model,
    subsample,
)


class AutoGluonModel(Model):
    def __init__(
        self,
        config_space: Dict,
        max_fit_samples: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        max_fit_time: Optional[float] = None,
    ):
        super(AutoGluonModel, self).__init__(
            random_state=random_state, config_space=config_space
        )
        self.max_fit_samples = max_fit_samples
        self.max_fit_time = max_fit_time

    def fit(self, X: pd.DataFrame, y: np.array):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, y_train = subsample(
            X_train,
            y_train,
            max_samples=self.max_fit_samples,
        )

        label = "y"
        df = X_train.copy()
        df[label] = y_train
        self.model = TabularPredictor(label=label).fit(df, time_limit=self.max_fit_time)
        self.sigma_train = np.mean(np.abs(self.model.predict(X_train).values - y_test))
        self.sigma_val = np.mean(np.abs(self.model.predict(X_test).values - y_test))
        # logging.warning(f"residual with {len(X)} samples: {self.sigma_val}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.array, np.array]:
        mu_pred = self.model.predict(X).values
        sigma_pred = np.ones_like(mu_pred) * self.sigma_val
        return mu_pred, sigma_pred
