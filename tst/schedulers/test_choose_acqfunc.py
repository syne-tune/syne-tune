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
from math import fabs

import pytest

from syne_tune.config_space import uniform, randint, choice
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import (
    create_tuning_job_state,
)
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import encode_state
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
    LCBAcquisitionFunction,
)


def _gp_searcher_with_state(gp_searcher, state):
    gp_searcher_state = gp_searcher.get_state()
    gp_searcher_state["state"] = encode_state(state)
    return gp_searcher.clone_from_state(gp_searcher_state)


# Uses code from examples/launch_standalone_bayesian_optimization.py. This is
# because we'd like to test whether the correct acquisition function object is
# created under realistic conditions.
@pytest.mark.skip("too long for CI")
def test_choose_acquisition_function():
    # toy example of 3 hp's
    config_space = {
        "hp_1": uniform(-5.0, 5.0),
        "hp_2": randint(-5, 5),
        "hp_3": choice(["a", "b", "c"]),
    }
    hp_ranges = make_hyperparameter_ranges(config_space)
    state = create_tuning_job_state(
        hp_ranges=hp_ranges,
        cand_tuples=[
            (-3.0, -4, "a"),
            (2.2, -3, "b"),
            (-4.9, -1, "b"),
            (-1.9, -1, "c"),
            (-3.5, 3, "a"),
        ],
        metrics=[dictionarize_objective(x) for x in (15.0, 27.0, 13.0, 39.0, 35.0)],
    )
    # Default should be EI
    gp_searcher = _gp_searcher_with_state(
        gp_searcher=GPFIFOSearcher(
            state.hp_ranges.config_space,
            metric="objective",
            debug_log=False,
        ),
        state=state,
    )
    acq_func = gp_searcher.acquisition_class(
        predictor=gp_searcher.state_transformer.fit()
    )
    assert isinstance(acq_func, EIAcquisitionFunction)
    # LCB with default kappa
    gp_searcher = _gp_searcher_with_state(
        gp_searcher=GPFIFOSearcher(
            state.hp_ranges.config_space,
            metric="objective",
            debug_log=False,
            acq_function="lcb",
        ),
        state=state,
    )
    acq_func = gp_searcher.acquisition_class(
        predictor=gp_searcher.state_transformer.fit()
    )
    assert isinstance(acq_func, LCBAcquisitionFunction)
    assert fabs(acq_func.kappa - 1.0) < 1e-8
    gp_searcher = _gp_searcher_with_state(
        gp_searcher=GPFIFOSearcher(
            state.hp_ranges.config_space,
            metric="objective",
            debug_log=False,
            acq_function="lcb",
            acq_function_kwargs={"kappa": 2.0},
        ),
        state=state,
    )
    acq_func = gp_searcher.acquisition_class(
        predictor=gp_searcher.state_transformer.fit()
    )
    assert isinstance(acq_func, LCBAcquisitionFunction)
    assert fabs(acq_func.kappa - 2.0) < 1e-8
