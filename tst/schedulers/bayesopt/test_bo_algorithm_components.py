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
import pytest
import numpy as np

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.config_space import uniform, choice, randint
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import (
    create_tuning_job_state,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    RandomFromSetCandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList


@pytest.fixture(scope="function")
def tuning_job_state():
    hp_ranges1 = make_hyperparameter_ranges(
        {"a1_hp_1": uniform(-5.0, 5.0), "a1_hp_2": choice(["a", "b", "c"])}
    )
    X1 = [(-3.0, "a"), (-1.9, "c"), (-3.5, "a")]
    Y1 = [dictionarize_objective(y) for y in (1.0, 2.0, 0.3)]
    hp_ranges2 = make_hyperparameter_ranges(
        {"a1_hp_1": uniform(-5.0, 5.0), "a1_hp_2": randint(-5, 5)}
    )
    X2 = [(-1.9, -1), (-3.5, 3)]
    Y2 = [dictionarize_objective(y) for y in (0.0, 2.0)]
    return {
        "algo-1": create_tuning_job_state(
            hp_ranges=hp_ranges1, cand_tuples=X1, metrics=Y1
        ),
        "algo-2": create_tuning_job_state(
            hp_ranges=hp_ranges2, cand_tuples=X2, metrics=Y2
        ),
    }


@pytest.fixture(scope="function")
def tuning_job_sub_state():
    hp_ranges = make_hyperparameter_ranges(dict())
    return TuningJobState.empty_state(hp_ranges)


BASE_SET = [
    {"a": 0.5, "b": "a"},
    {"a": 0.4, "b": "b"},
    {"a": 0.1, "b": "c"},
    {"a": 0.51, "b": "a"},
    {"a": 0.39, "b": "b"},
    {"a": 0.0, "b": "c"},
    {"a": 0.99, "b": "c"},
    {"a": 0.2, "b": "a"},
    {"a": 0.25, "b": "b"},
    {"a": 0.25, "b": "c"},
]


COMBINATIONS = [
    (
        15,
        [],
        10,
        list(range(10)),
    ),
    (
        15,
        [{"a": 0.51, "b": "a"}, {"a": 0.25, "b": "c"}],
        8,
        [0, 1, 2, 4, 5, 6, 7, 8],
    ),
    (
        8,
        [],
        8,
        None,
    ),
    (
        8,
        [{"a": 0.51, "b": "a"}, {"a": 0.25, "b": "c"}],
        8,
        [0, 1, 2, 4, 5, 6, 7, 8],
    ),
    (
        1,
        [
            {"a": 0.4, "b": "b"},
            {"a": 0.1, "b": "c"},
            {"a": 0.51, "b": "a"},
            {"a": 0.39, "b": "b"},
            {"a": 0.0, "b": "c"},
            {"a": 0.99, "b": "c"},
            {"a": 0.2, "b": "a"},
            {"a": 0.25, "b": "b"},
            {"a": 0.25, "b": "c"},
        ],
        1,
        [0],
    ),
]


@pytest.mark.parametrize("num_cands, excl_list, num_ret, pos_returned", COMBINATIONS)
def test_random_from_set_candidate_generator(
    num_cands, excl_list, num_ret, pos_returned
):
    random_seed = 31415927
    config_space = {
        "a": uniform(0.0, 1.0),
        "b": choice(["a", "b", "c"]),
    }
    hp_ranges = make_hyperparameter_ranges(config_space)
    random_state = np.random.RandomState(random_seed)
    random_generator = RandomFromSetCandidateGenerator(
        base_set=BASE_SET,
        random_state=random_state,
    )
    if excl_list:
        exclusion_list = ExclusionList(hp_ranges)
        for c in excl_list:
            exclusion_list.add(c)
    else:
        exclusion_list = None
    configs = random_generator.generate_candidates_en_bulk(
        num_cands, exclusion_list=exclusion_list
    )
    assert len(configs) == num_ret
    if pos_returned is not None:
        assert set(pos_returned) == random_generator.pos_returned
    else:
        pos_returned = random_generator.pos_returned
        assert len(pos_returned) == len(configs)
        configs_ms = set(hp_ranges.config_to_match_string(c) for c in configs)
        posret_ms = set(
            hp_ranges.config_to_match_string(BASE_SET[pos]) for pos in pos_returned
        )
        assert configs_ms == posret_ms
