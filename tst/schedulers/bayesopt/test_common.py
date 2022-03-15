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
from typing import List, Set, Tuple
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration, dictionarize_objective
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import ExclusionList, generate_unique_candidates
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import RepeatedCandidateGenerator
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.config_space import randint, choice
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import create_tuning_job_state, create_exclusion_set


@pytest.fixture(scope='function')
def hp_ranges():
    return make_hyperparameter_ranges({
        'hp1': randint(0, 200),
        'hp2': choice(['a', 'b', 'c'])})


@pytest.mark.parametrize('observed_data,failed_tuples,pending_tuples,expected', [
    ([], [], [], set()),
    ([((123, 'a'), 9.87)], [], [], {'hp1:123,hp2:0'}),
    ([], [(123, 'a')], [], {'hp1:123,hp2:0'}),
    ([], [], [(123, 'a')], {'hp1:123,hp2:0'}),
    ([((1, 'a'), 9.87)], [(2, 'b')], [(3, 'c')],
     {'hp1:1,hp2:0', 'hp1:2,hp2:1', 'hp1:3,hp2:2'})
])
def test_compute_blacklisted_candidates(
        hp_ranges: HyperparameterRanges,
        observed_data: List[Tuple],
        failed_tuples: List[Tuple],
        pending_tuples: List[Tuple],
        expected: Set[str]):
    if observed_data:
        cand_tuples, metrics = zip(*observed_data)
    else:
        cand_tuples = []
        metrics = []
    if metrics:
        metrics = [dictionarize_objective(y) for y in metrics]
    state = create_tuning_job_state(
        hp_ranges, cand_tuples=cand_tuples, metrics=metrics,
        pending_tuples=pending_tuples, failed_tuples=failed_tuples)
    actual = ExclusionList(state)
    assert set(expected) == actual.excl_set


def _assert_no_duplicates(
        candidates: List[Configuration], hp_ranges: HyperparameterRanges):
    cands_tpl = [hp_ranges.config_to_match_string(x) for x in candidates]
    assert len(candidates) == len(set(cands_tpl))


@pytest.mark.parametrize('num_unique_candidates,num_requested_candidates', [
    (5, 10),
    (15, 10)
])
def test_generate_unique_candidates(num_unique_candidates, num_requested_candidates):
    generator = RepeatedCandidateGenerator(num_unique_candidates)
    hp_ranges = generator.hp_ranges
    exclusion_candidates = create_exclusion_set([], hp_ranges)
    candidates = generate_unique_candidates(
        candidates_generator=generator,
        num_candidates=num_requested_candidates,
        exclusion_candidates=exclusion_candidates)
    assert len(candidates) == min(num_unique_candidates, num_requested_candidates)
    _assert_no_duplicates(candidates, hp_ranges)

    # introduce excluded candidates, simply take a few already unique
    size_excluded = len(candidates) // 2
    excluded = list(candidates)[:size_excluded]
    exclusion_candidates = create_exclusion_set(
        excluded, generator.hp_ranges, is_dict=True)
    candidates = generate_unique_candidates(
        candidates_generator=generator,
        num_candidates=num_requested_candidates,
        exclusion_candidates=exclusion_candidates)

    # total unique candidates are adjusted by the number of excluded candidates which are unique too due to set()
    assert len(candidates) == min(num_unique_candidates - len(excluded), num_requested_candidates)
    _assert_no_duplicates(candidates, hp_ranges)
