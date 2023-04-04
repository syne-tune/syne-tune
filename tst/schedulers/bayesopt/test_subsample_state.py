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
from typing import Tuple, List, Optional, Dict, Any
import pytest
from operator import itemgetter
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state_multi_fidelity import (
    cap_size_tuning_job_state,
    _extract_observations,
    _create_trials_evaluations,
    sparsify_tuning_job_state,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state_single_fidelity import (
    _extract_observations as _extract_observations_singlefid,
    _create_trials_evaluations as _create_trials_evaluations_singlefid,
    cap_size_tuning_job_state as cap_size_tuning_job_state_singlefid,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.config_space import uniform, lograndint
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.utils.successive_halving import (
    successive_halving_rung_levels,
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


def _data_from_state(state: TuningJobState) -> List[Tuple[int, int]]:
    return [(i, r) for i, r, _ in _extract_observations(state.trials_evaluations)]


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
    result = set(_data_from_state(new_state))
    assert len(result) <= max_size
    assert set(result_part).issubset(
        result
    ), f"data = {data}\nresult_part = {result_part}\nmax_size = {max_size}"


def _partition_wrt_rung_levels(
    data: List[Tuple[int, int]], rung_levels: List[int]
) -> List[List[Tuple[int, int]]]:
    r"""
    Partitions ``data`` into parts according to :math:`r_j < r \le r_{j+1}`,
    where :math:`[r_j]` is ``rung_levels``.
    """
    partition = []
    padded_rung_levels = [-1] + rung_levels
    sum_sizes = 0
    for r_low, r_high in zip(padded_rung_levels[:-1], padded_rung_levels[1:]):
        sub_data = [x for x in data if r_low < x[1] <= r_high]
        partition.append(sub_data)
        sum_sizes += len(sub_data)
    assert sum_sizes == len(data)  # Sanity check
    return partition


def _sparsify_at_most_one_per_trial(
    data: List[Tuple[int, int]], target_size: Optional[int]
) -> List[Tuple[int, int]]:
    """
    Filters ``data`` so that for every trial ID :math:`i` in ``data``, only
    the entry with the largest resource :math:`r` is retained. If ``target_size``
    is not ``None``, filtering is done in descending :math:`r` order, and is
    stopped once the remaining data has size ``target_size``.
    """
    return_size = len(data)
    assert target_size is None or return_size > target_size, (return_size, target_size)
    trials_covered = set()
    new_data = []
    sorted_data = sorted(data, key=itemgetter(1, 0), reverse=True)
    for pos, elem in enumerate(sorted_data):
        trial_id = elem[0]
        if trial_id not in trials_covered:
            new_data.append(elem)
            trials_covered.add(trial_id)
        else:
            return_size -= 1
            if return_size == target_size:
                break
    new_data.extend(sorted_data[(pos + 1) :])
    len_new_data = len(new_data)
    assert return_size == len_new_data, (return_size, len_new_data)
    return new_data


def _sparsify_data(
    data: List[Tuple[int, int]], max_size: int, rung_levels: List[int]
) -> List[Tuple[int, int]]:
    """
    Does the same as
    :func:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state.sparsify_tuning_job_state`,
    but using different code.
    """
    total_size = len(data)
    if total_size <= max_size:
        return data
    partition = _partition_wrt_rung_levels(data, rung_levels)
    new_partition = partition.copy()  # Replaces ``partition``
    for pos, sub_data in reversed(list(enumerate(partition))):
        if sub_data:
            subdata_size = len(sub_data)
            remaining_size = total_size - subdata_size
            if remaining_size <= max_size:
                target_size = max_size - remaining_size
            else:
                target_size = None
            new_sub_data = _sparsify_at_most_one_per_trial(sub_data, target_size)
            new_partition[pos] = new_sub_data
            new_subdata_size = len(new_sub_data)
            total_size -= subdata_size - new_subdata_size
            if target_size is not None and new_subdata_size == target_size:
                break  # Reached ``max_size``, so we are done
    return [elem for sub_data in new_partition for elem in sub_data]


def _error_message_comparison(
    data1: List[Tuple[int, int]],
    data2: List[Tuple[int, int]],
) -> str:
    parts = [
        f"{x}   {y}"
        for x, y in zip(
            sorted(data1, key=itemgetter(1, 0)), sorted(data2, key=itemgetter(1, 0))
        )
    ]
    return "\n".join(parts)


def _random_data(
    random_state: RandomState,
    r_min: Optional[int] = None,
    r_step: Optional[int] = None,
) -> Dict[str, Any]:
    # Create data at random
    num_trials = random_state.randint(low=1, high=30)
    if r_min is None:
        r_min = random_state.randint(low=1, high=4)
    r_max = r_min + random_state.randint(low=1, high=200)
    domain = lograndint(lower=r_min, upper=r_max)
    if r_step is None:
        r_step = random_state.randint(low=1, high=4)
    data = []
    for i in range(num_trials):
        r_top = domain.sample(random_state=random_state)
        data.extend((i, r) for r in range(r_min, r_top + 1, r_step))
    random_state.shuffle(data)
    data_size = len(data)
    if iter == 0 or data_size < 2:
        max_size = data_size + 1
    else:
        max_size = random_state.randint(low=max(data_size // 5, 1), high=data_size)
    # Compare outcomes
    reduction_factor = random_state.randint(low=2, high=5)
    print(
        f"num_trials = {num_trials}\n"
        f"r_min = {r_min}, r_max = {r_max}, r_step = {r_step}\n"
        f"data_size = {data_size}, max_size = {max_size}, "
        f"reduction_factor = {reduction_factor}"
    )
    return {
        "data": data,
        "r_min": r_min,
        "r_max": r_max,
        "max_size": max_size,
        "reduction_factor": reduction_factor,
    }


def test_sparsify_tuning_job_state():
    """
    Tests whether
    :func:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state.sparsify_tuning_job_state`
    does the same as a somewhat different implementation.
    """
    random_seed = 31415123
    random_state = RandomState(random_seed)
    config_space = {"lr": uniform(0, 1)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    num_iterations = 100
    for iter in range(num_iterations):
        # Create data at random
        config = _random_data(random_state)
        max_size = config["max_size"]
        data = config["data"]
        r_max = config["r_max"]
        # Compare outcomes
        new_state = sparsify_tuning_job_state(
            state=_state_from_data(data, hp_ranges),
            max_size=max_size,
            grace_period=config["r_min"],
            reduction_factor=config["reduction_factor"],
        )
        result = _data_from_state(new_state)
        len_result = len(result)
        print(f"len(result) = {len_result}")
        assert len_result >= min(max_size, len(data))
        rung_levels = successive_halving_rung_levels(
            rung_levels=None,
            grace_period=config["r_min"],
            reduction_factor=config["reduction_factor"],
            rung_increment=None,
            max_t=r_max,
        ) + [r_max]
        result_compare = _sparsify_data(data, max_size, rung_levels)
        print(f"len(result_compare) = {len(result_compare)}")
        assert set(result) == set(result_compare), _error_message_comparison(
            result, result_compare
        )


def _sparsify_data_simple(
    data: List[Tuple[int, int]], max_size: int, rung_levels: List[int]
) -> List[Tuple[int, int]]:
    """
    Different to :func:`_sparsify_data` above, data for trial :math:`i` is
    given for resource levels :math:`1, 2, \dots, r_i`.
    """
    data_size = len(data)
    if data_size <= max_size:
        return data
    r_min, r_max = rung_levels[0], rung_levels[-1]
    new_data = []
    done_max_of_trial = set()
    for resource in range(r_max, r_min - 1, -1):
        at_rung_level = resource in rung_levels
        trials_at_resource = sorted([i for i, r in data if r == resource], reverse=True)
        for pos, trial_id in enumerate(trials_at_resource):
            if at_rung_level or trial_id not in done_max_of_trial:
                new_data.append((trial_id, resource))
                done_max_of_trial.add(trial_id)
            else:
                data_size -= 1
                if data_size == max_size:
                    new_data.extend(
                        [(i, resource) for i in trials_at_resource[(pos + 1) :]]
                    )
                    break
        if data_size == max_size:
            new_data.extend([elem for elem in data if elem[1] < resource])
            assert len(new_data) == max_size, (len(new_data), max_size)
            break
    return new_data


def test_sparsify_tuning_job_state_simple():
    """
    Constructs datasets with dense targets. For this, the result of
    :func:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state.sparsify_tuning_job_state`
    is easy to determine.
    """
    random_seed = 31415123
    random_state = RandomState(random_seed)
    config_space = {"lr": uniform(0, 1)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    num_iterations = 100
    for iter in range(num_iterations):
        # Create data at random
        config = _random_data(random_state, r_min=1, r_step=1)
        max_size = config["max_size"]
        data = config["data"]
        r_max = config["r_max"]
        # Compare outcomes
        new_state = sparsify_tuning_job_state(
            state=_state_from_data(data, hp_ranges),
            max_size=max_size,
            grace_period=1,
            reduction_factor=config["reduction_factor"],
        )
        result = _data_from_state(new_state)
        len_result = len(result)
        print(f"len(result) = {len_result}")
        assert len_result >= min(max_size, len(data))
        rung_levels = successive_halving_rung_levels(
            rung_levels=None,
            grace_period=1,
            reduction_factor=config["reduction_factor"],
            rung_increment=None,
            max_t=r_max,
        ) + [r_max]
        result_compare = _sparsify_data_simple(data, max_size, rung_levels)
        print(f"len(result_compare) = {len(result_compare)}")
        assert set(result) == set(result_compare), _error_message_comparison(
            result, result_compare
        )


def _state_from_data_singlefid(
    data: List[Tuple[int, float]], hp_ranges: HyperparameterRanges
) -> TuningJobState:
    trial_ids = set(i for i, _ in data)
    config_for_trial = {str(trial_id): dict() for trial_id in trial_ids}
    return TuningJobState(
        hp_ranges=hp_ranges,
        config_for_trial=config_for_trial,
        trials_evaluations=_create_trials_evaluations_singlefid(data),
        failed_trials=[],
        pending_evaluations=[],
    )


def _data_from_state_singlefid(state: TuningJobState) -> List[Tuple[int, float]]:
    return _extract_observations_singlefid(state.trials_evaluations)


test_cases_singlefid = [
    (
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        [(5, 0), (6, 1)],
        6,
        1 / 3,
        "min",
    ),
    (
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        [(3, 16), (4, 10)],
        6,
        1 / 3,
        "max",
    ),
    (
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        [(5, 0)],
        2,
        1 / 2,
        "min",
    ),
    (
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        [(5, 0), (6, 1), (2, 2), (8, 2.5)],
        8,
        1 / 2,
        "min",
    ),
    (
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        [
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 16),
            (4, 10),
            (5, 0),
            (6, 1),
            (7, 8),
            (8, 2.5),
        ],
        9,
        1 / 3,
        "min",
    ),
]


@pytest.mark.parametrize(
    "data, result_part, max_size, top_fraction, mode",
    test_cases_singlefid,
)
def test_cap_size_tuning_job_state_singlefid(
    data, result_part, max_size, top_fraction, mode
):
    random_seed = 31415123
    random_state = RandomState(random_seed)
    config_space = {"lr": uniform(0, 1)}
    hp_ranges = make_hyperparameter_ranges(config_space)

    new_state = cap_size_tuning_job_state_singlefid(
        state=_state_from_data_singlefid(data, hp_ranges),
        max_size=max_size,
        mode=mode,
        top_fraction=top_fraction,
        random_state=random_state,
    )
    result = set(_data_from_state_singlefid(new_state))
    assert len(result) <= max_size
    assert set(result_part).issubset(
        result
    ), f"data = {data}\nresult_part = {result_part}\nmax_size = {max_size}\nresult = {result}"
    assert result.issubset(
        set(data)
    ), f"data = {data}\nresult_part = {result_part}\nmax_size = {max_size}\nresult = {result}"
