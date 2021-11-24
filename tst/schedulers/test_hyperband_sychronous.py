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
from typing import List, Tuple
import numpy as np

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import \
    SynchronousHyperbandBracket
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager \
    import SynchronousHyperbandBracketManager
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import \
    SynchronousHyperbandRungSystem


def _trial_ids(lst):
    return [x[0] for x in lst]


def _assert_next_slots(
        bracket: SynchronousHyperbandBracket,
        rungs: List[Tuple[int, int]], current_rung: int,
        current_pos: int, results: List[List[Tuple[int, float]]]):
    trials, milestone = bracket.next_slots()
    rung_size, level = rungs[current_rung]
    assert milestone == level
    sz = rung_size - current_pos
    if current_rung == 0:
        expected_trials = [None] * sz
    else:
        expected_trials = _trial_ids(
            results[current_rung])[current_pos:]
    assert trials == expected_trials


def test_hyperband_bracket():
    rungs = [(9, 1), (4, 3), (1, 9)]
    results = [
        [(0, 3.), (1, 5.), (2, 1.), (3, 4.), (4, 9.), (5, 6.), (6, 2.), (7, 7.), (8, 8.)],
        [(2, 3.1), (6, 3.0), (0, 2.9), (3, 3.0)],
        [(0, 1.)],
    ]
    steps = [2, 3, 1, 3, 2, 1, 1]
    current_rung = 0
    current_pos = 0
    bracket = SynchronousHyperbandBracket(rungs, mode='min')
    for num_res in steps:
        _assert_next_slots(
            bracket, rungs, current_rung, current_pos, results)
        result = results[current_rung]
        is_completed = bracket.on_results(
            result[current_pos:(current_pos + num_res)])
        current_pos += num_res
        assert is_completed == (current_pos >= rungs[current_rung][0])
        if is_completed:
            current_rung += 1
            current_pos = 0


# Runs Hyperband for some number of iterations, checking that no assertions
# are raised
def test_hyperband_bracket_manager():
    random_seed = 31415927
    random_state = np.random.RandomState(random_seed)

    bracket_rungs = SynchronousHyperbandRungSystem.geometric(
        min_resource=2, max_resource=200, reduction_factor=3,
        num_brackets=6)
    bracket_manager = SynchronousHyperbandBracketManager(
        bracket_rungs, mode='min')
    num_jobs = 4
    num_steps = 5000
    next_trial_id = 0
    for step in range(num_steps):
        next_jobs = bracket_manager.next_jobs(num_jobs)
        num_assigned = sum([len(trials) for _, (trials, _) in next_jobs])
        assert num_assigned == num_jobs
        bracket_to_results = dict()
        for bracket_id, (trials, milestone) in next_jobs:
            sz = len(trials)
            if trials[0] is None:
                # Trials to be started from scratch
                trials = list(range(next_trial_id, next_trial_id + sz))
                next_trial_id += sz
            metric_vals = list(random_state.random(sz))
            bracket_to_results[bracket_id] = list(zip(trials, metric_vals))
        bracket_manager.on_results(bracket_to_results)
