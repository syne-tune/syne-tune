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
from typing import List, Callable, Optional, Dict, Tuple
import numpy as np
from sklearn.linear_model import RidgeCV
from enum import IntEnum

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
from syne_tune.optimizer.schedulers.searchers.searcher import impute_points_to_evaluate

__all__ = [
    "LinearCostModel",
    "MLPLinearCostModel",
    "FixedLayersMLPCostModel",
    "NASBench201LinearCostModel",
    "BiasOnlyLinearCostModel",
]


class LinearCostModel(CostModel):
    """
    Deterministic cost model where both c0(x) and c1(x) are linear models of
    the form

        c0(x) = np.dot(features0(x), weights0),
        c1(x) = np.dot(features1(x), weights1)

    The feature maps features0, features1 are supplied by subclasses.
    The weights are fit by ridge regression, using scikit.learn RidgeCV, the
    regularization constant is set by LOO cross-validation.

    """

    def __init__(self):
        self.weights0 = None
        self.weights1 = None

    @property
    def cost_metric_name(self) -> str:
        return INTERNAL_COST_NAME

    def feature_matrices(
        self, candidates: List[Configuration]
    ) -> (np.ndarray, np.ndarray):
        """
        Has to be supplied by subclasses

        :param candidates: List of n candidate configs (non-extended)
        :return: Feature matrices features0 (n, dim0), features1 (n, dim1)
        """
        raise NotImplementedError

    def update(self, state: TuningJobState):
        # Compile feature matrix and targets for linear regression problem
        configs = [
            state.config_for_trial[ev.trial_id] for ev in state.trials_evaluations
        ]
        features0, features1 = self.feature_matrices(configs)
        dim0 = features0.shape[1]
        feature_parts = []
        cost_parts = []
        for feature0, feature1, ev in zip(
            features0, features1, state.trials_evaluations
        ):
            metric_vals = ev.metrics.get(self.cost_metric_name)
            if metric_vals is not None:
                assert isinstance(metric_vals, dict)
                resource_values, cost_values = zip(*metric_vals.items())
                resource_values = np.array(resource_values, dtype=np.float64).reshape(
                    (-1, 1)
                )
                feature0 = feature0.astype(np.float64, copy=False).reshape((1, -1))
                feature1 = feature1.astype(np.float64, copy=False).reshape((1, -1))
                feature_parts.append(
                    np.concatenate(
                        (
                            np.broadcast_to(
                                feature0, (resource_values.size, feature0.size)
                            ),
                            resource_values * feature1,
                        ),
                        axis=1,
                    )
                )
                cost_parts.append(
                    np.array(cost_values, dtype=np.float64).reshape((-1, 1))
                )
        features = np.vstack(feature_parts)
        targets = np.vstack(cost_parts).reshape((-1,))
        assert features.shape[0] == targets.size
        assert features.shape[1] == dim0 + features1.shape[1]
        # Fit with RidgeCV, where alpha is selected by LOO cross-validation
        predictor = RidgeCV(alphas=np.exp(np.arange(-4, 5)), fit_intercept=False).fit(
            features, targets
        )
        self.weights0 = predictor.coef_[:dim0].reshape((-1, 1))
        self.weights1 = predictor.coef_[dim0:].reshape((-1, 1))
        self.alpha = predictor.alpha_

    def sample_joint(self, candidates: List[Configuration]) -> List[CostValue]:
        assert self.weights0 is not None, "Must call 'update' before 'sample_joint'"
        features0, features1 = self.feature_matrices(candidates)
        c0_vals = np.matmul(features0, self.weights0).reshape((-1,))
        c1_vals = np.matmul(features1, self.weights1).reshape((-1,))
        return [CostValue(c0, c1) for c0, c1 in zip(c0_vals, c1_vals)]


class BiasOnlyLinearCostModel(LinearCostModel):
    """
    Simple baseline: features0(x) = [1], features1(x) = [1]

    """

    def __init__(self):
        super().__init__()

    def feature_matrices(
        self, candidates: List[Configuration]
    ) -> (np.ndarray, np.ndarray):
        one_feats = np.ones((len(candidates), 1))
        return one_feats, one_feats


class MLPLinearCostModel(LinearCostModel):
    """
    Deterministic linear cost model for multi-layer perceptron.

    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: Callable[[dict], int],
        hidden_layer_width: Callable[[dict, int], int],
        batch_size: Callable[[dict], int],
        bs_exponent: Optional[float] = None,
        extra_mlp: bool = False,
        c0_mlp_feature: bool = False,
        expected_hidden_layer_width: Optional[Callable[[int], float]] = None,
    ):
        """
        If config is a HP configuration, num_hidden_layers(config) is the
        number of hidden layers, hidden_layer_width(config, layer) is the
        number of units in hidden layer layer (0-based), batch_size(config)
        is the batch size.

        If expected_hidden_layer_width is given, it maps layer (0-based) to
        expected layer width under random sampling. In this case, all MLP
        features are normalized to expected value 1 under random sampling
        (but ignoring bs_exponent if != 1).
        Note: If needed, we could incorporate bs_exponent in general. If
        batch_size was uniform between a and b:
            E[ power(bs, bs_exp - 1) ] =
            (power(b, bs_exp) - power(a, bs_exp)) / (bs_exp * (b - a))

        :param num_inputs: Number of input nodes
        :param num_outputs: Number of output nodes
        :param num_hidden_layers: See above
        :param hidden_layer_width: See above
        :param batch_size: See above
        :param bs_exponent: Main MLP feature is multiplied by
            power(batch_size, bs_exponent - 1)
        :param extra_mlp: Add additional "linear" MLP feature to c_1?
        :param c0_mlp_feature: Use main MLP feature in c_0 as well?
        :param expected_hidden_layer_width: See above
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.batch_size = batch_size
        if bs_exponent is not None:
            self.bs_exponent = bs_exponent
        else:
            self.bs_exponent = 1
        self.extra_mlp = extra_mlp
        self.c0_mlp_feature = c0_mlp_feature
        self.expected_hidden_layer_width = expected_hidden_layer_width

    def feature_matrices(
        self, candidates: List[Configuration]
    ) -> (np.ndarray, np.ndarray):
        features1_1 = []
        features1_2 = []
        for config in candidates:
            value = self._mlp_feature(config)
            if self.bs_exponent != 1:
                bs = self.batch_size(config)
                value *= np.power(bs, self.bs_exponent - 1)
            features1_1.append(value)
            if self.extra_mlp:
                features1_2.append(self._mlp_feature2(config))
        ones_vec = np.ones((len(features1_1), 1))
        features1_1 = np.array(features1_1).reshape((-1, 1))
        features1_tpl = (ones_vec, features1_1)
        if self.extra_mlp:
            features1_tpl += (np.array(features1_2).reshape((-1, 1)),)
        features1 = np.concatenate(features1_tpl, axis=1)
        if self.c0_mlp_feature:
            features0 = np.concatenate((ones_vec, features1_1), axis=1)
        else:
            features0 = ones_vec
        return features0, features1

    def _mlp_feature(self, config: Configuration) -> float:
        layers = range(self.num_hidden_layers(config))
        width_list = [self.hidden_layer_width(config, layer) for layer in layers]
        if self.expected_hidden_layer_width is None:
            norm_const = 1
        else:
            norm_const = self._sum_of_prod(
                [self.expected_hidden_layer_width(layer) for layer in layers]
            )
        return self._sum_of_prod(width_list) / norm_const

    def _sum_of_prod(self, lst):
        return sum(
            x * y for x, y in zip([self.num_inputs] + lst, lst + [self.num_outputs])
        )

    def _mlp_feature2(self, config: Configuration) -> float:
        layers = range(self.num_hidden_layers(config))
        width_list = [self.hidden_layer_width(config, layer) for layer in layers]
        if self.expected_hidden_layer_width is None:
            norm_const = 1
        else:
            norm_const = sum(
                self.expected_hidden_layer_width(layer) for layer in layers
            )
        return sum(width_list) / norm_const


class FixedLayersMLPCostModel(MLPLinearCostModel):
    """
    Linear cost model for MLP with num_hidden_layers hidden layers.

    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_units_keys: List[str] = None,
        bs_exponent: Optional[float] = None,
        extra_mlp: bool = False,
        c0_mlp_feature: bool = False,
        expected_hidden_layer_width: Optional[Callable[[int], float]] = None,
    ):
        if num_units_keys is None:
            num_units_keys = ["n_units_1", "n_units_2"]
        num_hidden_layers = len(num_units_keys)

        def hidden_layer_width(config, layer):
            return int(config[num_units_keys[layer]])

        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_hidden_layers=lambda config: num_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            batch_size=lambda config: int(config["batch_size"]),
            bs_exponent=bs_exponent,
            extra_mlp=extra_mlp,
            c0_mlp_feature=c0_mlp_feature,
            expected_hidden_layer_width=expected_hidden_layer_width,
        )

    @staticmethod
    def get_expected_hidden_layer_width(config_space: Dict, num_units_keys: List[str]):
        """
        Constructs expected_hidden_layer_width function from the training
        evaluation function.
        Works because `impute_points_to_evaluate` imputes with the expected
        value under random sampling.

        :param config_space: Configuration space
        :param num_units_keys: Keys into `config_space` for number of
            units of different layers
        :return: expected_hidden_layer_width, exp_vals
        """
        default_config = impute_points_to_evaluate(None, config_space)[0]
        exp_vals = [default_config[k] for k in num_units_keys]

        def expected_hidden_layer_width(x):
            return exp_vals[x]

        return expected_hidden_layer_width, exp_vals


class NASBench201LinearCostModel(LinearCostModel):
    """
    Deterministic linear cost model for NASBench201.

    The cell graph is:
        node1 = x0(node0)
        node2 = x1(node0) + x2(node1)
        node3 = x3(node0) + x4(node1) + x5(node2)

    """

    class Op(IntEnum):
        SKIP_CONNECT = 0
        NONE = 1
        NOR_CONV_1x1 = 2
        NOR_CONV_3x3 = 3
        AVG_POOL_3x3 = 4

    def __init__(
        self,
        config_keys: Tuple[str, ...],
        map_config_values: Dict[str, int],
        conv_separate_features: bool,
        count_sum: bool,
    ):
        """
        `config_keys` contains attribute names of x0, ..., x5 in a config, in
        this ordering. `map_config_values` maps values in the config (for
        fields corresponding to x0, ..., x5) to entries of `Op`.

        :param config_keys: See above
        :param map_config_values: See above
        :param conv_separate_features: If True, we use separate features for
            nor_conv_1x1, nor_conv_3x3 (c1 has 4 features). Otherwise, these
            two are captured by a single features (c1 has 3 features)
        :param count_sum: If True, we use an additional feature for pointwise
            sum operators inside a cell (there are between 0 and 3)

        """
        super().__init__()
        self._config_keys = config_keys
        self._map_config_values = map_config_values
        self.conv_separate_features = conv_separate_features
        self.count_sum = count_sum

    def _translate(self, config: Configuration) -> List[int]:
        return [self._map_config_values[config[name]] for name in self._config_keys]

    def feature_matrices(
        self, candidates: List[Configuration]
    ) -> (np.ndarray, np.ndarray):
        features1_1 = []
        features1_2 = []
        features1_3 = []  # If conv_separate_features
        features1_4 = []  # If count_sum
        none_val = NASBench201LinearCostModel.Op.NONE
        for config in candidates:
            operators = self._translate(config)
            # Certain NONE (or "zeroize") values imply other NONE values:
            # x0 = > x2 and x4
            # x1 and x2 = > x5
            if operators[0] == none_val:
                operators[2] = none_val
                operators[4] = none_val
            if operators[1] == none_val and operators[2] == none_val:
                operators[5] = none_val
            n_conv1, n_conv3, n_apool = map(
                sum,
                zip(
                    *(
                        (
                            x == NASBench201LinearCostModel.Op.NOR_CONV_1x1,
                            x == NASBench201LinearCostModel.Op.NOR_CONV_3x3,
                            x == NASBench201LinearCostModel.Op.AVG_POOL_3x3,
                        )
                        for x in operators
                    )
                ),
            )
            features1_1.append((5 / 6) * n_apool)
            if self.conv_separate_features:
                features1_2.append((5 / 6) * n_conv1)
                features1_3.append((5 / 6) * n_conv3)
            else:
                features1_2.append((n_conv1 + 9 * n_conv3) / 12)
            if self.count_sum:
                features1_4.append(
                    (25 / 76)
                    * (
                        (operators[1] != none_val) * (operators[2] != none_val)
                        + (operators[3] != none_val)
                        + (operators[4] != none_val)
                        + (operators[5] != none_val)
                    )
                )

        ones_vec = np.ones((len(features1_1), 1))
        features1_1 = np.array(features1_1).reshape((-1, 1))
        features1_2 = np.array(features1_2).reshape((-1, 1))
        features1_tpl = (ones_vec, features1_1, features1_2)
        if self.conv_separate_features:
            features1_tpl += (np.array(features1_3).reshape((-1, 1)),)
        if self.count_sum:
            features1_tpl += (np.array(features1_4).reshape((-1, 1)),)
        features1 = np.concatenate(features1_tpl, axis=1)
        return ones_vec, features1
