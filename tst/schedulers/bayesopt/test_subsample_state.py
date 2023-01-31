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
from typing import Tuple, List
import pytest
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.subsample_state import (
    cap_size_tuning_job_state,
    _extract_observations,
    _create_trials_evaluations,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
    HyperparameterRanges,
)


def _state_from_data(
    data: List[Tuple[int, int]], hp_ranges: HyperparameterRanges
) -> TuningJobState:
    trial_ids = set(i for i, _ in data)
    config_for_trial = {str(trial_id): dict() for trial_id in trial_ids}
    return TuningJobState(
        hp_ranges=hp_ranges,
        config_for_trial=config_for_trial,
        trials_evaluations=_create_trials_evaluations([(i, r, 0) for i, r in data]),
        failed_trials=[],
        pending_evaluations=[],
    )


test_cases = [
    (
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        [
            (1, 1),
            (3, 1),
            (5, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        14,
    ),
    (
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        [
            (1, 1),
            (1, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        6,
    ),
    (
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        [
            (1, 21),
        ],
        1,
    ),
    (
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        18,
    ),
    (
        [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (3, 3),
            (5, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        [
            (1, 1),
            (7, 1),
            (1, 3),
            (8, 3),
            (7, 3),
            (8, 9),
            (1, 9),
            (7, 9),
            (1, 21),
        ],
        10,
    ),
]


@pytest.mark.parametrize(
    "data, result_part, max_size",
    test_cases,
)
def test_cap_size_tuning_job_state(data, result_part, max_size):
    random_seed = 31415123
    random_state = RandomState(random_seed)
    config_space = {"lr": uniform(0, 1)}
    hp_ranges = make_hyperparameter_ranges(config_space)

    new_state = cap_size_tuning_job_state(
        state=_state_from_data(data, hp_ranges),
        max_size=max_size,
        random_state=random_state,
    )
    result = set(
        (i, r) for i, r, _ in _extract_observations(new_state.trials_evaluations)
    )
    assert len(result) <= max_size
    assert set(result_part).issubset(
        result
    ), f"data = {data}\nresult_part = {result_part}\nmax_size = {max_size}"
