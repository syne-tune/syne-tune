from typing import Optional, Dict
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import syne_tune.config_space as sp
from benchmarking.blackbox_repository.blackbox import Blackbox


class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return X[self.names]


class BlackboxSurrogate(Blackbox):

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            configuration_space: Dict,
            fidelity_space: Optional[Dict] = None,
            fidelity_values: Optional[np.array] = None,
            surrogate=KNeighborsRegressor(n_neighbors=1),
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
        :param surrogate: the model that is fitted to predict objectives given any configuration.
        Possible examples: KNeighborsRegressor(n_neighbors=1), MLPRegressor() or any estimator obeying Scikit-learn API.
        The model is fit on top of pipeline that applies basic feature-processing to convert rows in X to vectors.
        We use the configuration_space hyperparameters types to deduce the types of columns in X (for instance
        CategoricalHyperparameter are one-hot encoded).
        :param name:
        """
        super(BlackboxSurrogate, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=y.columns,
        )
        assert len(X) == len(y)
        # todo other types of assert with configuration_space, objective_names, ...
        self.X = X
        self.y = y
        self.surrogate = surrogate
        self.fit_surrogate(surrogate)
        self.name = name
        self._fidelity_values = fidelity_values

    @property
    def fidelity_values(self) -> np.array:
        return self._fidelity_values

    def fit_surrogate(self, surrogate=KNeighborsRegressor(n_neighbors=1)) -> Blackbox:
        """
        Fits a surrogate model to a blackbox.
        :param surrogate: fits the model and apply the model transformation when evaluating a
        blackbox configuration. Possible example: KNeighborsRegressor(n_neighbors=1), MLPRegressor() or any estimator
        obeying Scikit-learn API.
        """
        self.surrogate = surrogate

        # gets hyperparameters types, categorical for CategoricalHyperparameter, numeric for everything else
        numeric = []
        categorical = []

        if self.fidelity_space is not None:
            surrogate_hps = dict()
            surrogate_hps.update(self.configuration_space)
            surrogate_hps.update(self.fidelity_space)
        else:
            surrogate_hps = self.configuration_space

        for hp_name, hp in surrogate_hps.items():
            if isinstance(hp, sp.Categorical):
                categorical.append(hp_name)
            else:
                numeric.append(hp_name)

        # apply transformation to columns according its type
        print(f"fitting surrogate predicting {self.y.columns} given numerical cols {numeric} and "
              f"categorical cols {categorical}")

        # builds a pipeline that standardize numeric features and one-hot categorical ones before applying
        # the surrogate model
        features_union = []
        if len(categorical) > 0:
            features_union.append(('categorical', make_pipeline(
                Columns(names=categorical), OneHotEncoder(sparse=False, handle_unknown='ignore'))))
        if len(numeric) > 0:
            features_union.append(('numeric', make_pipeline(Columns(names=numeric), StandardScaler())))

        self.surrogate_pipeline = Pipeline([
            ("features", FeatureUnion(features_union)),
            ('model', surrogate)
        ])

        self.surrogate_pipeline.fit(
            X=self.X,
            y=self.y
        )
        return self

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict[str, float]:
        surrogate_input = configuration.copy()
        if fidelity is not None or self.fidelity_values is None:
            if fidelity is not None:
                surrogate_input.update(fidelity)
            # use the surrogate model for prediction
            prediction = self.surrogate_pipeline.predict(pd.DataFrame([surrogate_input]))

            # converts the returned nd-array with shape (1, num_metrics) to the list of objectives values
            prediction = prediction.reshape(-1).tolist()

            # convert prediction to dictionary
            return dict(zip(self.objectives_names, prediction))
        else:
            # when no fidelity is given and a fidelity space exists, we return all fidelities
            # we construct a input dataframe with all fidelity for the configuration given to call the transformer
            # at once which is more efficient due to vectorization
            surrogate_input_df = pd.DataFrame([surrogate_input] * len(self.fidelity_values))
            surrogate_input_df[next(iter(self.fidelity_space.keys()))] = self.fidelity_values
            objectives_values = self.surrogate_pipeline.predict(surrogate_input_df)
            return objectives_values


def add_surrogate(blackbox: Blackbox, surrogate=KNeighborsRegressor(n_neighbors=1)):
    """
    Fits a blackbox surrogates that can be evaluated anywhere, which can be useful for supporting
    interpolation/extrapolation.
    :param blackbox: the blackbox must implement `hyperparame`ter_objectives_values`
    so that input/output are passed to estimate the model, see `BlackboxOffline` or `BlackboxTabular
    :param surrogate: the model that is fitted to predict objectives given any configuration.
    Possible examples: KNeighborsRegressor(n_neighbors=1), MLPRegressor() or any estimator obeying Scikit-learn API.
    The model is fit on top of pipeline that applies basic feature-processing to convert rows in X to vectors.
    We use the configuration_space hyperparameters types to deduce the types of columns in X (for instance
    CategoricalHyperparameter are one-hot encoded).
    :return: a blackbox where the output is obtained through the fitted surrogate
    """
    X, y = blackbox.hyperparameter_objectives_values()
    return BlackboxSurrogate(
        X=X,
        y=y,
        configuration_space=blackbox.configuration_space,
        fidelity_space=blackbox.fidelity_space,
        fidelity_values=blackbox.fidelity_values,
        surrogate=surrogate,
    )
