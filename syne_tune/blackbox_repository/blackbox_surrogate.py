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
from typing import Optional
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from syne_tune.config_space import Categorical
from syne_tune.blackbox_repository.blackbox import Blackbox


class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return X[self.names]


def _default_surrogate(surrogate):
    if surrogate is not None:
        return surrogate
    else:
        return KNeighborsRegressor(n_neighbors=1)


class BlackboxSurrogate(Blackbox):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        configuration_space: dict,
        fidelity_space: Optional[dict] = None,
        fidelity_values: Optional[np.array] = None,
        surrogate=None,
        max_fit_samples: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """
        Fits a blackbox surrogates that can be evaluated anywhere, which can be useful for supporting
        interpolation/extrapolation. To wrap an existing blackbox with a surrogate estimator, use `add_surrogate`
        which automatically extract X, y matrices from available blackbox evaluations.
        :param X: dataframe containing hyperparameters values, columns should be the ones in configuration_space
        and fidelity_space
        :param y: dataframe containing objectives values
        :param configuration_space:
        :param fidelity_space:
        :param surrogate: the model that is fitted to predict objectives given any configuration, default to
        KNeighborsRegressor(n_neighbors=1).
        Possible examples: KNeighborsRegressor(n_neighbors=1), MLPRegressor() or any estimator obeying Scikit-learn API.
        The model is fit on top of pipeline that applies basic feature-processing to convert rows in X to vectors.
        We use the configuration_space hyperparameters types to deduce the types of columns in X (for instance
        CategoricalHyperparameter are one-hot encoded).
        :param max_fit_samples: maximum number of samples to be fed to the surrogate estimator, if the more data points
        than this number are passed, then they are subsampled without replacement.
        :param name:
        """
        super(BlackboxSurrogate, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=y.columns,
        )
        assert len(X) == len(y)
        # todo other types of assert with configuration_space, objective_names, ...
        self.surrogate = _default_surrogate(surrogate)
        self.max_fit_samples = max_fit_samples
        self.fit_surrogate(
            X=X, y=y, surrogate=surrogate, max_samples=self.max_fit_samples
        )
        self.name = name
        self._fidelity_values = fidelity_values

    @property
    def fidelity_values(self) -> np.array:
        return self._fidelity_values

    @staticmethod
    def make_model_pipeline(configuration_space, fidelity_space, model):
        # gets hyperparameters types, categorical for CategoricalHyperparameter, numeric for everything else
        numeric = []
        categorical = []

        if fidelity_space is not None:
            surrogate_hps = dict()
            surrogate_hps.update(configuration_space)
            surrogate_hps.update(fidelity_space)
        else:
            surrogate_hps = configuration_space

        for hp_name, hp in surrogate_hps.items():
            if isinstance(hp, Categorical):
                categorical.append(hp_name)
            else:
                numeric.append(hp_name)

        # builds a pipeline that standardize numeric features and one-hot categorical ones before applying
        # the surrogate model
        features_union = []
        if len(categorical) > 0:
            features_union.append(
                (
                    "categorical",
                    make_pipeline(
                        Columns(names=categorical),
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    ),
                )
            )
        if len(numeric) > 0:
            features_union.append(
                ("numeric", make_pipeline(Columns(names=numeric), StandardScaler()))
            )

        return Pipeline(
            [
                ("features", FeatureUnion(features_union)),
                ("standard scaler", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )

    def fit_surrogate(
        self, X, y, surrogate=None, max_samples: Optional[int] = None
    ) -> Blackbox:
        """
        Fits a surrogate model to a blackbox.
        :param surrogate: fits the model and apply the model transformation when evaluating a
        blackbox configuration. Possible example: KNeighborsRegressor(n_neighbors=1), MLPRegressor() or any estimator
        obeying Scikit-learn API.
        """
        self.surrogate = _default_surrogate(surrogate)

        self.surrogate_pipeline = self.make_model_pipeline(
            configuration_space=self.configuration_space,
            fidelity_space=self.fidelity_space,
            model=surrogate,
        )
        # todo would be nicer to have this in the feature pipeline
        if max_samples is not None and max_samples < len(X):
            random_indices = np.random.permutation(len(X))[:max_samples]
            self.surrogate_pipeline.fit(
                X=X.loc[random_indices], y=y.loc[random_indices]
            )
        else:
            self.surrogate_pipeline.fit(X=X, y=y)
        return self

    def _objective_function(
        self,
        configuration: dict,
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> dict:
        surrogate_input = configuration.copy()
        if fidelity is not None or self.fidelity_values is None:
            if fidelity is not None:
                surrogate_input.update(fidelity)
            # use the surrogate model for prediction
            prediction = self.surrogate_pipeline.predict(
                pd.DataFrame([surrogate_input])
            )

            # converts the returned nd-array with shape (1, num_metrics) to the list of objectives values
            prediction = prediction.reshape(-1).tolist()

            # convert prediction to dictionary
            return dict(zip(self.objectives_names, prediction))
        else:
            # when no fidelity is given and a fidelity space exists, we return all fidelities
            # we construct a input dataframe with all fidelity for the configuration given to call the transformer
            # at once which is more efficient due to vectorization
            surrogate_input_df = pd.DataFrame(
                [surrogate_input] * len(self.fidelity_values)
            )
            surrogate_input_df[
                next(iter(self.fidelity_space.keys()))
            ] = self.fidelity_values
            objectives_values = self.surrogate_pipeline.predict(surrogate_input_df)
            return objectives_values


def add_surrogate(blackbox: Blackbox, surrogate=None, configuration_space=None):
    """
    Fits a blackbox surrogates that can be evaluated anywhere, which can be useful
    for supporting interpolation/extrapolation.
    :param blackbox: the blackbox must implement `hyperparameter_objectives_values`
        so that input/output are passed to estimate the model, see `BlackboxOffline`
        or `BlackboxTabular
    :param surrogate: the model that is fitted to predict objectives given any
        configuration. Possible examples: `KNeighborsRegressor(n_neighbors=1)`,
        `MLPRegressor()` or any estimator obeying Scikit-learn API.
        The model is fit on top of pipeline that applies basic feature-processing
        to convert rows in X to vectors. We use `config_space` to deduce the types
        of columns in X (categorical parameters are 1-hot encoded).
    :param configuration_space: configuration space for the resulting blackbox surrogate.
        The default is `blackbox.configuration_space`. But note that if `blackbox`
        is tabular, the domains in `blackbox.configuration_space` are typically
        categorical even for numerical parameters.
    :return: a blackbox where the output is obtained through the fitted surrogate
    """
    surrogate = _default_surrogate(surrogate)
    if configuration_space is None:
        configuration_space = blackbox.configuration_space
    X, y = blackbox.hyperparameter_objectives_values()
    return BlackboxSurrogate(
        X=X,
        y=y,
        configuration_space=configuration_space,
        fidelity_space=blackbox.fidelity_space,
        fidelity_values=blackbox.fidelity_values,
        surrogate=surrogate,
    )
