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
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging

from syne_tune.config_space import Categorical
from syne_tune.blackbox_repository.blackbox import (
    Blackbox,
    ObjectiveFunctionResult,
)
from syne_tune.blackbox_repository.blackbox_offline import BlackboxOffline

logger = logging.getLogger(__name__)


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
    """
    Fits a blackbox surrogates that can be evaluated anywhere, which can be
    useful for supporting interpolation/extrapolation. To wrap an existing
    blackbox with a surrogate estimator, use :func:`add_surrogate` which
    automatically extract ``X``, ``y`` matrices from available blackbox evaluations.

    The surrogate regression model is provided by ``surrogate``, it has to
    conform to the scikit-learn fit-predict API. If ``predict_curves`` is ``True``,
    the model maps features of the configuration to the whole curve over
    fidelities, separate for each metric and seed. This has several advantages.
    First, predictions are consistent: if all curves in the data respect a certain
    property which is retained under convex combinations, predictions have this
    property as well (examples: positivity, monotonicity). This is important for
    ``elapsed_time`` metrics. The regression models are also fairly compact, and
    prediction is fast, ``max_fit_samples`` is normally not needed.

    If ``predict_curves`` is ``False,`` the model maps features from configuration and
    fidelity to metric values (univariate regression). In this case, properties
    like monotonicity are not retained. Also, training can take long and the
    trained models can be large.

    This difference only matters if there are fidelities. Otherwise, regression
    is always univariate.

    If ``num_seeds`` is given, we maintain different surrogate models for each
    seed. Otherwise, a single surrogate model is fit to data across all seeds.

    If ``fit_differences`` is given, it contains names of objectives which
    are cumulative sums. For these objectives, the ``y`` data is transformed
    to finite differences before fitting the model. This is recommended for
    ``elapsed_time`` objectives. This feature only matters if there are
    fidelities.

    Additional arguments on top of parent class
    :class:`~syne_tune.blackbox_repository.Blackbox`:

    :param X: dataframe containing hyperparameters values. Shape is
        ``(num_seeds * num_evals, num_hps)`` if ``predict_curves`` is ``True``,
        ``(num_fidelities * num_seeds * num_evals, num_hps)`` otherwise
    :param y: dataframe containing objectives values. Shape is
        ``(num_seeds * num_evals, num_fidelities * num_objectives)`` if
        ``predict_curves`` is ``True``, and
        ``(num_fidelities * num_seeds * num_evals, num_objectives)`` otherwise
    :param surrogate: the model that is fitted to predict objectives given any
        configuration, default to KNeighborsRegressor(n_neighbors=1). If
        ``predict_curves`` is ``True``, this must be multi-variate regression, i.e.
        accept target matrices in ``fit``, where columns correspond to fidelities.
        Regression models from scikit-learn allow for that.
        Possible examples: :code:`KNeighborsRegressor(n_neighbors=1)`,
        :code:`MLPRegressor()` or any estimator obeying Scikit-learn API.
        The model is fit on top of pipeline that applies basic feature-processing
        to convert rows in ``X`` to vectors. We use the configuration_space
        hyperparameters types to deduce the types of columns in ``X`` (for instance,
        :class:`~syne_tune.config_space.Categorical` values are one-hot encoded).
    :param predict_curves: See above. Default is ``False`` (backwards compatible)
    :param num_seeds: See above
    :param fit_differences: See above
    :param max_fit_samples: maximum number of samples to be fed to the surrogate
        estimator, if the more data points than this number are passed, then they
        are subsampled without replacement. If ``num_seeds`` is used, this is a
        limit on the data per seed
    :param name:
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        configuration_space: Dict[str, Any],
        objectives_names: List[str],
        fidelity_space: Optional[dict] = None,
        fidelity_values: Optional[np.array] = None,
        surrogate=None,
        predict_curves: bool = False,
        num_seeds: int = 1,
        fit_differences: Optional[List[str]] = None,
        max_fit_samples: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super(BlackboxSurrogate, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_names=objectives_names,
        )
        assert len(X) == len(y)
        if self.fidelity_space is None or fidelity_values is None:
            # Always univariate regression then
            predict_curves = False
            fit_differences = None
        self._assert_shapes(
            X, y, predict_curves, fidelity_values, num_seeds, len(objectives_names)
        )
        # todo other types of assert with configuration_space, objective_names, ...
        self.surrogate = _default_surrogate(surrogate)
        self.surrogate_pipeline = None
        self.max_fit_samples = max_fit_samples
        self.predict_curves = predict_curves
        if fit_differences is None:
            fit_differences = []
        else:
            fit_differences = [objectives_names.index(name) for name in fit_differences]
        self.fit_differences = fit_differences
        self.name = name
        self._fidelity_values = fidelity_values
        self.num_seeds = num_seeds
        self.fit_surrogate(X=X, y=y)

    @staticmethod
    def _assert_shapes(
        X: pd.DataFrame,
        y: pd.DataFrame,
        predict_curves: bool,
        fidelity_values: Optional[np.ndarray],
        num_seeds: Optional[int],
        num_objectives: int,
    ):
        assert X.ndim == 2 and y.ndim == 2
        num_rows = X.shape[0]
        assert num_rows == y.shape[0]
        assert num_seeds >= 1 and num_rows % num_seeds == 0
        if predict_curves:
            num_fidelities = len(fidelity_values)
            assert (
                y.shape[1] == num_objectives * num_fidelities
            ), f"y.shape[1] = {y.shape[1]} != {num_fidelities} * {num_objectives}"
        elif fidelity_values is not None:
            num_fidelities = len(fidelity_values)
            num_evalsxseeds = num_rows // num_fidelities
            assert (
                num_evalsxseeds >= 1 and num_rows == num_evalsxseeds * num_fidelities
            ), f"X.shape[0] = {num_rows} != {num_fidelities} * {num_evalsxseeds}"
            assert num_evalsxseeds >= num_seeds
            assert (
                y.shape[1] == num_objectives
            ), f"y.shape[1] = {y.shape[1]} != {num_objectives}"

    @property
    def fidelity_values(self) -> Optional[np.array]:
        return self._fidelity_values

    @property
    def num_fidelities(self) -> int:
        if self.fidelity_values is not None:
            num_fidelities = len(self.fidelity_values)
        else:
            num_fidelities = 1
        return num_fidelities

    @staticmethod
    def make_model_pipeline(
        configuration_space, fidelity_space, model, predict_curves=False
    ):
        """Create feature pipeline for scikit-learn model

        :param configuration_space: Configuration space
        :param fidelity_space: Fidelity space
        :param model: Scikit-learn model
        :param predict_curves: Predict full curves?
        :return: Feature pipeline
        """
        # gets hyperparameters types, categorical for CategoricalHyperparameter, numeric for everything else
        numeric = []
        categorical = []

        if fidelity_space is not None and not predict_curves:
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

    def _data_for_seeds(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.num_seeds == 1:
            return [X], [y]
        else:
            dim1 = self.num_seeds
            dim0 = 1 if self.predict_curves else self.num_fidelities
            dim2 = X.shape[0] // (dim1 * dim0)
            assert dim2 >= 1 and X.shape[0] == dim0 * dim1 * dim2
            Xs = []
            ys = []
            for seed in range(self.num_seeds):
                index = np.ravel(
                    dim2 * (dim1 * np.arange(dim0).reshape((-1, 1)) + seed)
                    + np.arange(dim2).reshape((1, -1))
                )
                Xs.append(X.iloc[index])
                ys.append(y.iloc[index])
            return Xs, ys

    def _fidelity_spacing(self) -> (np.ndarray, bool):
        subtract_vec = np.concatenate((np.zeros(1), self.fidelity_values[:-1]))
        spacing = self.fidelity_values - subtract_vec
        is_contiguous = np.all(spacing == 1)
        return spacing, is_contiguous

    def _transform_to_finite_differences(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Note: ``fidelity_values`` need not be contiguous (1, 2, 3, ...). We use
        generalized weighted finite differences to account for that.

        :param y: Original data (cumulative sum format)
        :return: Transformed data (finite differences)
        """
        num_fidelities = self.num_fidelities
        num_objectives = len(self.objectives_names)
        if num_fidelities > 1 and self.fit_differences:
            spacing, is_contiguous = self._fidelity_spacing()
            if self.predict_curves:
                y_shape = (-1, num_fidelities, num_objectives)
            else:
                y_shape = (num_fidelities, -1, num_objectives)
            y_data = y.to_numpy()
            y_orig_shape = y_data.shape
            y_data = y_data.reshape(y_shape)
            rng_fid_plus = np.arange(1, num_fidelities)
            rng_fid_minus = np.arange(num_fidelities - 1)
            for objective_pos in self.fit_differences:
                if self.predict_curves:
                    y_temp = y_data[:, rng_fid_minus, objective_pos].copy()
                    y_data[:, rng_fid_plus, objective_pos] -= y_temp
                    if not is_contiguous:
                        y_data[:, :, objective_pos] /= spacing.reshape((1, -1))
                else:
                    y_temp = y_data[rng_fid_minus, :, objective_pos].copy()
                    y_data[rng_fid_plus, :, objective_pos] -= y_temp
                    if not is_contiguous:
                        y_data[:, :, objective_pos] /= spacing.reshape((-1, 1))
            return pd.DataFrame(data=y_data.reshape(y_orig_shape))
        else:
            return y

    def _transform_from_finite_differences(self, prediction: np.ndarray) -> np.ndarray:
        """
        Note: ``fidelity_values`` need not be contiguous (``1, 2, 3, ...``). We use
        generalized weighted finite differences to account for that.

        :param prediction: Shape ``(num_fidelities, num_objectives)``
        :return:
        """
        num_fidelities = self.num_fidelities
        if num_fidelities > 1 and self.fit_differences:
            spacing, is_contiguous = self._fidelity_spacing()
            if is_contiguous:
                spacing = 1
            for objective_pos in self.fit_differences:
                prediction_new = np.cumsum(prediction[:, objective_pos] * spacing)
                prediction[:, objective_pos] = prediction_new
            return prediction
        else:
            return prediction

    def fit_surrogate(self, X: pd.DataFrame, y: pd.DataFrame) -> Blackbox:
        """
        Fits a surrogate model to data from a blackbox. Here, the targets ``y`` can
        be a matrix with the number of columns equal to the number of fidelity
        values (the ``predict_curves = True`` case).
        """
        self.surrogate_pipeline = [
            self.make_model_pipeline(
                configuration_space=self.configuration_space,
                fidelity_space=self.fidelity_space,
                model=self.surrogate,
                predict_curves=self.predict_curves,
            )
            for _ in range(self.num_seeds)
        ]
        y = self._transform_to_finite_differences(y)
        Xs, ys = self._data_for_seeds(X, y)
        for pipeline, features, targets in zip(self.surrogate_pipeline, Xs, ys):
            # todo would be nicer to have this in the feature pipeline
            num_data = len(features)
            if self.max_fit_samples is not None and self.max_fit_samples < num_data:
                random_indices = np.random.permutation(num_data)[: self.max_fit_samples]
                features = features.loc[random_indices]
                targets = targets.loc[random_indices]
            pipeline.fit(X=features, y=targets)
        return self

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> ObjectiveFunctionResult:
        if seed is None:
            seed = np.random.randint(0, self.num_seeds)
        else:
            assert (
                0 <= seed < self.num_seeds
            ), f"seed = {seed}, must be in [0, {self.num_seeds - 1}]"
        surrogate_input = configuration.copy()
        single_fidelity = fidelity is not None
        do_fit_diffs = len(self.fit_differences) > 0
        if self.fidelity_values is not None:
            fidelity_attr = self.fidelity_name()
        else:
            fidelity_attr = None
        if not self.predict_curves:
            # Univariate regression, where fidelity is an input
            if (not do_fit_diffs) and (single_fidelity or self.fidelity_values is None):
                if single_fidelity:
                    surrogate_input.update(fidelity)
                # use the surrogate model for prediction
                prediction = self.surrogate_pipeline[seed].predict(
                    pd.DataFrame([surrogate_input])
                )
                # converts the returned nd-array with shape (1, num_metrics)
                # to the list of objectives values
                prediction = prediction.reshape(-1).tolist()
                # convert prediction to dictionary
                prediction = dict(zip(self.objectives_names, prediction))
            else:
                # when no fidelity is given and a fidelity space exists, we
                # return all fidelities
                # we construct a input dataframe with all fidelity for the
                # configuration given to call the transformer at once which
                # is more efficient due to vectorization
                surrogate_input_df = pd.DataFrame(
                    [surrogate_input] * self.num_fidelities
                )
                surrogate_input_df[fidelity_attr] = self.fidelity_values
                prediction = self._transform_from_finite_differences(
                    self.surrogate_pipeline[seed].predict(surrogate_input_df)
                )
            extract_fidelity = do_fit_diffs and single_fidelity
        else:
            # Multivariate regression
            prediction = self.surrogate_pipeline[seed].predict(
                pd.DataFrame([surrogate_input])
            )
            prediction = self._transform_from_finite_differences(
                prediction.reshape((self.num_fidelities, -1))
            )
            extract_fidelity = single_fidelity

        if extract_fidelity:
            assert self.fidelity_values is not None, "blackbox has no fidelities"
            # If there are several fidelity values, pick the first
            fidelity = list(fidelity.values())[0]
            ind = np.where(self.fidelity_values == fidelity)
            assert ind, f"fidelity {fidelity} not among {self.fidelity_values}"
            ind = ind[0]
            prediction = dict(zip(self.objectives_names, prediction[ind]))
        return prediction

    def hyperparameter_objectives_values(
        self, predict_curves: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("This is a surrogate already!")


def add_surrogate(
    blackbox: Blackbox,
    surrogate=None,
    configuration_space: Optional[dict] = None,
    predict_curves: Optional[bool] = None,
    separate_seeds: bool = False,
    fit_differences: Optional[List[str]] = None,
):
    """
    Fits a blackbox surrogates that can be evaluated anywhere, which can be useful
    for supporting interpolation/extrapolation.

    :param blackbox: the blackbox must implement
        :meth:`~syne_tune.blackbox_repository.Blackbox.hyperparameter_objectives_values`
        so that input/output are passed to estimate the model
    :param surrogate: the model that is fitted to predict objectives given any
        configuration. Possible examples: :code:`KNeighborsRegressor(n_neighbors=1)`,
        :code:`MLPRegressor()` or any estimator obeying Scikit-learn API.
        The model is fit on top of pipeline that applies basic feature-processing
        to convert rows in ``X`` to vectors. We use ``configuration_space`` to deduce
        the types of columns in ``X`` (categorical parameters are one-hot encoded).
    :param configuration_space: configuration space for the resulting blackbox
        surrogate. The default is ``blackbox.configuration_space``. But note that
        if ``blackbox`` is tabular, the domains in ``blackbox.configuration_space``
        are typically categorical even for numerical parameters.
    :param predict_curves: If True, the surrogate uses multivariate regression
        to predict metric curves over fidelities. If False, fidelity is used
        as input. The latter can lead to inconsistent predictions along
        fidelity and is typically more expensive.
        If not given, the default value is ``False`` if ``blackbox`` is of type
        :class:`~syne_tune.blackbox_repository.BlackboxOffline`, otherwise ``True``.
    :param separate_seeds: If ``True``, seeds in ``blackbox`` map to seeds in the
        surrogate blackbox, which fits different models to each seed. If ``False``,
        the data from ``blackbox`` is merged for all seeds, and the surrogate
        represents a single seed. The latter provides more data for the surrogate
        model to be fit, but the variation between seeds is lost in the
        surrogate. Defaults to ``False``.
    :param fit_differences: Names of objectives which are cumulative sums. For
        these objectives, the ``y`` data is transformed to finite differences
        before fitting the model. This is recommended for ``elapsed_time``
        objectives.
    :return: a blackbox where the output is obtained through the fitted surrogate
    """
    if configuration_space is None:
        configuration_space = blackbox.configuration_space
    if separate_seeds and blackbox.fidelity_values is not None:
        num_seeds = len(blackbox.fidelity_values)
    else:
        num_seeds = 1
    if predict_curves is None:
        # ``BlackboxOffline`` does not support True right now
        predict_curves = not isinstance(blackbox, BlackboxOffline)
    X, y = blackbox.hyperparameter_objectives_values(predict_curves)
    return BlackboxSurrogate(
        X=X,
        y=y,
        configuration_space=configuration_space,
        objectives_names=blackbox.objectives_names,
        fidelity_space=blackbox.fidelity_space,
        fidelity_values=blackbox.fidelity_values,
        surrogate=surrogate,
        predict_curves=predict_curves,
        num_seeds=num_seeds,
        fit_differences=fit_differences,
    )
