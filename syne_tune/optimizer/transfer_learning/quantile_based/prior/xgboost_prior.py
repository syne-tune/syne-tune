import numpy as np
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from syne_tune.optimizer.transfer_learning.quantile_based.prior import FunctionalPrior


class XGBoostPrior(FunctionalPrior):
    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            **xgb_kwargs,
    ):
        self.output_dim = y_train.shape[-1]
        self.estimators = [
            xgb.XGBRegressor(
                **xgb_kwargs,
            ) for i in range(self.output_dim)
        ]
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_train)

        for i in range(self.output_dim):
            self.estimators[i].fit(X, y_train[..., i])

    def predict(self, X):
        X = self.scaler.transform(X)
        mus = []
        for i in range(self.output_dim):
            mus.append(self.estimators[i].predict(X))
        mu = np.array(mus).T
        if self.output_dim == 1:
            # always return (num_samples, num_output_dim)
            mu = mu.reshape((-1, 1))
        sigma = np.ones_like(mu)
        return mu, sigma


if __name__ == '__main__':

    num_train_examples = 10000
    num_test_examples = num_train_examples
    dim = 2
    num_gradient_updates = 200
    lr = 1e-2

    def make_random_X_y(num_examples: int, dim: int, noise_std: float):
        X = np.random.rand(num_examples, dim)
        noise = np.random.normal(scale=noise_std, size=(num_examples, 1))
        y = X.sum(axis=-1, keepdims=True) + noise
        return X, y


    # test that parametric prior can recover a simple linear function for the mean
    noise_std = 0.01
    X_train, y_train = make_random_X_y(num_examples=num_train_examples, dim=dim, noise_std=noise_std)
    prior = XGBoostPrior(
        X_train=X_train,
        y_train=y_train,
        #num_gradient_updates=num_gradient_updates,
        #num_decays=2,
        # smaller network for UT speed
        #num_layers=2,
        #num_hidden=20,
        #dropout=0.0,
        #lr=lr
    )
    X_test, y_test = make_random_X_y(num_examples=num_test_examples, dim=dim, noise_std=noise_std)
    mu_pred, sigma_pred = prior.predict(X_test)

    mu_l1_error = np.abs(mu_pred - y_test).mean()
    print(mu_l1_error)
    assert mu_l1_error < 0.2
