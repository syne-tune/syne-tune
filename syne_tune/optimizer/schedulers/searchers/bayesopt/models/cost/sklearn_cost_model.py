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
from typing import List, Tuple, Callable
import numpy as np
import logging
from syne_tune.try_import import try_import_blackbox_repository_message

try:
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
    )
except ImportError:
    logging.info(try_import_blackbox_repository_message())
    raise
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.cost_model import (
    CostModel,
    CostValue,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_COST_NAME,
)

__all__ = ["ScikitLearnCostModel", "UnivariateSplineCostModel"]


class NonLinearCostModel(CostModel):
    """
    Deterministic cost model, where c0(x) = b0 (constant), and c1(x) is given
    by a nonlinear regression model specified in subclasses. Parameters are b0
    and those of the regression model. We use a simple algorithm to jointly fit
    b0 and c1(x).

    """

    def __init__(self):
        self.b0 = None
        self.regr_model = None
        self.hp_ranges = None

    @property
    def cost_metric_name(self) -> str:
        return INTERNAL_COST_NAME

    def transform_dataset(
        self, dataset: List[Tuple[Configuration, float]], num_data0: int, res_min: int
    ) -> dict:
        """
        Transforms dataset (see `_data_for_c1_regression`) into a dataset
        representation (dict), which is used as `kwargs` in `fit_regressor`.

        :param dataset:
        :param num_data0:
        :param res_min:
        :return: Used as kwargs in fit_regressor
        """
        raise NotImplementedError

    @staticmethod
    def fit_regressor(b0: float, **kwargs):
        """
        Given value for b0, fits regressor to dataset specified via kwargs
        (see `transform_dataset`). Returns the criterion function value for
        b0 as well as the fitted regression model.

        :param b0:
        :param kwargs:
        :return: fval, model
        """
        raise NotImplementedError

    def predict_c1_values(self, candidates: List[Configuration]):
        """
        :param candidates: Test configs
        :return: Corresponding c1 values
        """
        raise NotImplementedError

    def update(self, state: TuningJobState):
        self.hp_ranges = state.hp_ranges  # Needed in transform_dataset
        # Prepare data for fitting c1(x)
        dataset, num_data0, res_min, target_min = self._data_for_c1_regression(state)
        assert target_min > 0  # Sanity check
        data_kwargs = self.transform_dataset(dataset, num_data0, res_min)
        best = [None]  # Model corresponding to root
        # Since critfunc is not strictly well-defined, we need to
        # cache values for previous evals at the same b0. In
        # particular, this avoids "invalid bracket" errors when
        # brentq evaluates at the bracket ends.
        cf_cache = dict()

        def critfunc(b0):
            if b0 in cf_cache:
                fval = cf_cache[b0]
            else:
                fval, model = self.fit_regressor(b0, **data_kwargs)
                cf_cache[b0] = fval
                absfval = abs(fval)
                best_tup = best[0]
                if (best_tup is None) or (absfval < best_tup[0]):
                    best[0] = (absfval, b0, model)
            return fval

        # Root finding for b0
        atol = 1e-5
        ftol = 1e-8
        f_low = critfunc(0)
        if num_data0 < len(dataset) and f_low < -ftol:
            # f(0) < -ftol < 0
            f_high = critfunc(target_min)
            if f_high > ftol:
                # f(target_min) > ftol > 0: We have a bracket
                try:
                    brentq(critfunc, a=0, b=target_min, xtol=atol)
                except Exception:
                    # Use best evaluated until exception
                    pass
        _, self.b0, self.regr_model = best[0]

    def sample_joint(self, candidates: List[Configuration]) -> List[CostValue]:
        assert self.b0 is not None, "Must call 'update' before 'sample_joint'"
        c1_vals = self.predict_c1_values(candidates)
        c0_vals = np.full(len(c1_vals), self.b0)
        return [CostValue(c0, c1) for c0, c1 in zip(c0_vals, c1_vals)]

    def _data_for_c1_regression(self, state: TuningJobState):
        """
        Extracts `dataset` as list of (config, target) tuples. The first
        num_data0 records correspond to configs appearing only once in
        `state`, at the minimum resource level `res_min`.

        :param state: TuningJobState
        :return: dataset, num_data0, res_min, target_min
        """
        data_config = []
        for ev in state.trials_evaluations:
            metric_vals = ev.metrics[self.cost_metric_name]
            assert isinstance(metric_vals, dict)
            config = state.config_for_trial[ev.trial_id]
            data_config.append((config, list(metric_vals.items())))
        res_min = min(min(res for res, _ in tpls) for _, tpls in data_config)
        target_min = min(min(cost for _, cost in tpls) for _, tpls in data_config)
        # Split data into two parts (r = res_min, r > res_min),
        # compute transformed target values
        data_0, data_1 = [], []
        for config, targets in data_config:
            if len(targets) > 1:
                # config has >1 entry -> data_1
                targets = sorted(targets, key=lambda x: x[0])
                lst = [
                    (x1[1] - x2[1]) / (x1[0] - x2[0])
                    for x1, x2 in zip(targets[:-1], targets[1:])
                ]
                data_1.extend([(config, y) for y in lst])
            x = targets[0]
            assert x[0] == res_min, "config = {}, targets = {}".format(config, targets)
            data_0.append((config, x[1] / res_min))
        # Return dataset: data_0 comes before data_1
        num_data0 = len(data_0)
        return data_0 + data_1, num_data0, res_min, target_min


_supported_model_types = {"random_forest", "gradient_boosting"}


class ScikitLearnCostModel(NonLinearCostModel):
    """
    Deterministic cost model, where c0(x) = b0 (constant), and c1(x) is given
    by a scikit.learn (or scipy) regression model. Parameters are b0 and those
    of the regression model.

    """

    def __init__(self, model_type=None):
        """
        :param model_type: Regression model for c1(x)

        """
        if model_type is None:
            model_type = "random_forest"
        else:
            assert (
                model_type in _supported_model_types
            ), "model_type = '{}' not supported, must be in {}".format(
                model_type, _supported_model_types
            )
        super().__init__()
        self.model_type = model_type

    def transform_dataset(
        self, dataset: List[Tuple[Configuration, float]], num_data0: int, res_min: int
    ) -> dict:
        num_hps = len(self.hp_ranges)
        num_data = len(dataset)
        features = np.zeros((num_data, num_hps))
        targets = np.zeros(num_data)
        for i, (config, target) in enumerate(dataset):
            features[i, :] = self.hp_ranges.to_ndarray(config, categ_onehot=False)
            targets[i] = target
        return {
            "features": features,
            "targets": targets,
            "num_data0": num_data0,
            "res_min": res_min,
            "model_type": self.model_type,
        }

    @staticmethod
    def fit_regressor(b0: float, **kwargs):
        features = kwargs["features"]
        targets = kwargs["targets"]
        num_data0 = kwargs["num_data0"]
        res_min = kwargs["res_min"]
        _targets = targets.copy()
        _targets[:num_data0] -= b0 / res_min
        if kwargs["model_type"] == "random_forest":
            model = RandomForestRegressor(n_estimators=50)
        else:
            model = GradientBoostingRegressor()
        model.fit(features, _targets)
        # Compute root finding criterion for b0
        resvec = (
            model.predict(features[:num_data0]).reshape((-1,)) - targets[:num_data0]
        )
        crit_val = (np.sum(resvec) + b0 * num_data0 / res_min) / res_min
        return crit_val, model

    def predict_c1_values(self, candidates: List[Configuration]):
        features1 = self.hp_ranges.to_ndarray_matrix(candidates, categ_onehot=False)
        c1_vals = self.regr_model.predict(features1).reshape((-1,))
        return c1_vals


class UnivariateSplineCostModel(NonLinearCostModel):
    """
    Here, c1(x) is given by a univariate spline
    (scipy.optimize.UnivariateSpline), where a single scalar is extracted from
    x.

    In the second part of the dataset (pos >= num_data0), duplicate entries with
    the same config in dataset are grouped into one, using the mean as target
    value, and a weight equal to the number of duplicates. This still leaves
    duplicates in the overall dataset, one in data0, the other in data1, but
    spline smoothing can deal with this.

    """

    def __init__(
        self,
        scalar_attribute: Callable[[Configuration], float],
        input_range: Tuple[float, float],
        spline_degree: int = 3,
    ):
        """
        :param scalar_attribute: Maps config to scalar input attribute
        :param input_range: (lower, upper), range for input attribute
        :param spline_degree: Degree for smoothing spline, in 1, ..., 5
        """
        assert (
            spline_degree >= 1 and spline_degree <= 5
        ), "spline_degree = {} invalid, must be integer in [1, 5]".format(spline_degree)
        assert (
            len(input_range) == 2 and input_range[0] < input_range[1]
        ), "input_range = {} not valid range for input attribute"
        super().__init__()
        self.scalar_attribute = scalar_attribute
        self.input_range = input_range
        self.spline_degree = spline_degree

    def transform_dataset(
        self, dataset: List[Tuple[Configuration, float]], num_data0: int, res_min: int
    ) -> dict:
        # We combine duplicates in the second part of the dataset
        config_lst, target_lst = zip(*dataset[:num_data0])
        config_lst = list(config_lst)
        target_lst = list(target_lst)
        weight_lst = [1] * num_data0
        data_config = dict()
        for config, target in dataset[num_data0:]:
            config_key = self.hp_ranges.config_to_match_string(config)
            if config_key in data_config:
                data_config[config_key][1].append(target)
            else:
                data_config[config_key] = (config, [target])
        for config, targets in data_config.values():
            config_lst.append(config)
            target_lst.append(np.mean(targets))
            weight_lst.append(len(targets))
        # Create scalar features
        features = np.array([self.scalar_attribute(config) for config in config_lst])
        targets = np.array(target_lst)
        weights = np.array(weight_lst)
        return {
            "features": features,
            "targets": targets,
            "weights": weights,
            "num_data0": num_data0,
            "res_min": res_min,
            "input_range": self.input_range,
            "spline_degree": self.spline_degree,
        }

    @staticmethod
    def fit_regressor(b0: float, **kwargs):
        features = kwargs["features"]
        targets = kwargs["targets"]
        weights = kwargs["weights"]
        num_data0 = kwargs["num_data0"]
        res_min = kwargs["res_min"]
        input_range = kwargs["input_range"]
        spline_degree = min(kwargs["spline_degree"], targets.size - 1)
        _targets = targets.copy()
        _targets[:num_data0] -= b0 / res_min
        # Inputs must be in increasing order
        sort_ind = np.argsort(features)
        _features = features[sort_ind]
        _targets = _targets[sort_ind]
        _weights = weights[sort_ind]
        # Merge cases with equal inputs (UnivariateSpline does not work
        # with duplicate inputs)
        feature_lst = []
        target_lst = []
        weight_lst = []
        x = _features[0]
        wsum = _weights[0]
        y = wsum * _targets[0]
        sz = targets.size
        _features = np.insert(_features, sz, _features[-1] + 10)  # Guard
        for i in range(1, sz + 1):
            x_new = _features[i]
            if x_new == x:
                w_new = _weights[i]
                y += w_new * _targets[i]
                wsum += w_new
            else:
                feature_lst.append(x)
                target_lst.append(y / wsum)
                weight_lst.append(wsum)
                if i < sz:
                    x = x_new
                    wsum = _weights[i]
                    y = wsum * _targets[i]
        model = UnivariateSpline(
            x=feature_lst, y=target_lst, w=weight_lst, bbox=input_range, k=spline_degree
        )
        # Compute root finding criterion for b0
        resvec = model(features[:num_data0]).reshape((-1,)) - targets[:num_data0]
        crit_val = (np.sum(resvec) + b0 * num_data0 / res_min) / res_min
        return crit_val, model

    def predict_c1_values(self, candidates: List[Configuration]):
        features1 = np.array([self.scalar_attribute(config) for config in candidates])
        c1_vals = self.regr_model(features1).reshape((-1,))
        return c1_vals
