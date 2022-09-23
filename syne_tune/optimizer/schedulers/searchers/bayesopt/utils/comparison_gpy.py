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
from typing import List, Dict
import numpy as np
import copy

from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    dictionarize_objective,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    get_internal_candidate_evaluations,
)


class ThreeHumpCamel:
    @property
    def search_space(self):
        return [{"min": -5.0, "max": 5.0}, {"min": -5.0, "max": 5.0}]

    def evaluate(self, x1, x2):
        return 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2


def branin_function(x1, x2, r=6):
    return (
        (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - r) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )


class Branin:
    @property
    def search_space(self):
        return [{"min": -5.0, "max": 10.0}, {"min": 0.0, "max": 15.0}]

    def evaluate(self, x1, x2):
        return branin_function(x1, x2)


class BraninWithR(Branin):
    def __init__(self, r):
        self.r = r

    def evaluate(self, x1, x2):
        return branin_function(x1, x2, r=self.r)


class Ackley:
    @property
    def search_space(self):
        const = 32.768
        return [{"min": -const, "max": const}, {"min": -const, "max": const}]

    def evaluate(self, x1, x2):
        a = 20
        b = 0.2
        c = 2 * np.pi
        ssq = (x1**2) + (x2**2)
        scos = np.cos(c * x1) + np.cos(c * x2)
        return (
            -a * np.exp(-b * np.sqrt(0.5 * ssq)) - np.exp(0.5 * scos) + (a + np.exp(1))
        )


class SimpleQuadratic:
    @property
    def search_space(self):
        return [{"min": 0.0, "max": 1.0}, {"min": 0.0, "max": 1.0}]

    def evaluate(self, x1, x2):
        return 2 * (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2


def _decode_input(x, lim):
    mn, mx = lim["min"], lim["max"]
    return x * (mx - mn) + mn


def evaluate_blackbox(bb_func, inputs: np.ndarray) -> np.ndarray:
    num_dims = inputs.shape[1]
    input_list = []
    for x, lim in zip(np.split(inputs, num_dims, axis=1), bb_func.search_space):
        input_list.append(_decode_input(x, lim))
    return bb_func.evaluate(*input_list)


# NOTE: Inputs will always be in [0, 1] (so come in encoded form). They are
# only scaled to their native ranges (linearly) when evaluations of the
# blackbox are done. This avoids silly errors.
def sample_data(
    bb_cls, num_train: int, num_grid: int, expand_datadct: bool = True
) -> dict:
    bb_func = bb_cls()
    ss_limits = bb_func.search_space
    num_dims = len(ss_limits)
    # Sample training inputs
    train_inputs = np.random.uniform(low=0.0, high=1.0, size=(num_train, num_dims))
    # Training targets (currently, no noise is added)
    train_targets = evaluate_blackbox(bb_func, train_inputs).reshape((-1,))
    # Inputs for prediction (regular grid)
    grids = [np.linspace(0.0, 1.0, num_grid)] * num_dims
    grids2 = tuple(np.meshgrid(*grids))
    test_inputs = np.hstack([x.reshape(-1, 1) for x in grids2])
    # Also evaluate true function on grid
    true_targets = evaluate_blackbox(bb_func, test_inputs).reshape((-1,))
    data = {
        "ss_limits": ss_limits,
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "test_inputs": test_inputs,
        "grid_shape": grids2[0].shape,
        "true_targets": true_targets,
    }
    if expand_datadct:
        # Make sure that ours and GPy below receive exactly the same inputs
        data = expand_data(data)
    return data


def expand_data(data: dict) -> dict:
    """
    Appends derived entries to data dict, which have non-elementary types.
    """
    if "state" not in data:
        data = copy.copy(data)
        state = data_to_state(data)
        data_internal = get_internal_candidate_evaluations(
            state,
            active_metric=INTERNAL_METRIC_NAME,
            normalize_targets=True,
            num_fantasy_samples=20,
        )
        data["state"] = state
        data["train_inputs"] = data_internal.features
        data["train_targets_normalized"] = data_internal.targets
    return data


# Recall that inputs in data are encoded, so we have to decode them to their
# native ranges for `trials_evaluations`
def data_to_state(data: dict) -> TuningJobState:
    configs, cs = decode_inputs(data["train_inputs"], data["ss_limits"])
    config_for_trial = {
        str(trial_id): config for trial_id, config in enumerate(configs)
    }
    trials_evaluations = [
        TrialEvaluations(trial_id=str(trial_id), metrics=dictionarize_objective(y))
        for trial_id, y in enumerate(data["train_targets"])
    ]
    return TuningJobState(
        hp_ranges=make_hyperparameter_ranges(cs),
        config_for_trial=config_for_trial,
        trials_evaluations=trials_evaluations,
    )


def decode_inputs(inputs: np.ndarray, ss_limits) -> (List[Configuration], Dict):
    cs_names = [f"x{i}" for i in range(len(ss_limits))]
    cs = {
        name: uniform(lower=lims["min"], upper=lims["max"])
        for name, lims in zip(cs_names, ss_limits)
    }
    x_mult = []
    x_add = []
    for lim in ss_limits:
        mn, mx = lim["min"], lim["max"]
        x_mult.append(mx - mn)
        x_add.append(mn)
    x_mult = np.array(x_mult)
    x_add = np.array(x_add)
    configs = []
    for x in inputs:
        x_decoded = x * x_mult + x_add
        config_dct = dict(zip(cs_names, x_decoded))
        configs.append(config_dct)
    return configs, cs


def assert_equal_candidates(candidates1, candidates2, hp_ranges, decimal=5):
    inputs1 = hp_ranges.to_ndarray_matrix(candidates1)
    inputs2 = hp_ranges.to_ndarray_matrix(candidates2)
    np.testing.assert_almost_equal(inputs1, inputs2, decimal=decimal)


def assert_equal_randomstate(randomstate1, randomstate2):
    assert str(randomstate1.get_state()) == str(randomstate2.get_state())
