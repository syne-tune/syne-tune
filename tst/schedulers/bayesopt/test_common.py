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
    import CandidateEvaluation, PendingEvaluation, Configuration, dictionarize_objective
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import ExclusionList, generate_unique_candidates
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import RepeatedCandidateGenerator
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.search_space import uniform, randint, choice
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import tuples_to_configs, create_exclusion_set


@pytest.fixture(scope='function')
def hp_ranges():
    return make_hyperparameter_ranges({
        'hp1': randint(0, 200),
        'hp2': choice(['a', 'b', 'c'])})


@pytest.fixture(scope='function')
def multi_algo_state():
    def _candidate_evaluations(num, hp_ranges: HyperparameterRanges):
        return [
            CandidateEvaluation(
                candidate=hp_ranges.tuple_to_config((i,)),
                metrics=dictionarize_objective(float(i)))
            for i in range(num)]

    hp_ranges0 = make_hyperparameter_ranges({'a1_hp_1': uniform(-5.0, 5.0)})
    hp_ranges1 = make_hyperparameter_ranges(dict())

    return {
        '0': TuningJobState(
            hp_ranges=hp_ranges0,
            candidate_evaluations=_candidate_evaluations(2, hp_ranges0),
            failed_candidates=tuples_to_configs(
                [(i,) for i in range(3)], hp_ranges0),
            pending_evaluations=[
                PendingEvaluation(hp_ranges0.tuple_to_config((i,)))
                for i in range(100)]),
        '1': TuningJobState(
            hp_ranges=hp_ranges1,
            candidate_evaluations=_candidate_evaluations(5, hp_ranges1),
            failed_candidates=[],
            pending_evaluations=[]),
        '2': TuningJobState(
            hp_ranges=hp_ranges1,
            candidate_evaluations=_candidate_evaluations(3, hp_ranges1),
            failed_candidates=tuples_to_configs(
                [(i,) for i in range(10)], hp_ranges1),
            pending_evaluations=[
                PendingEvaluation(hp_ranges0.tuple_to_config((i,)))
                for i in range(1)]),
        '3': TuningJobState(
            hp_ranges=hp_ranges1,
            candidate_evaluations=_candidate_evaluations(6, hp_ranges1),
            failed_candidates=[],
            pending_evaluations=[]),
        '4': TuningJobState(
            hp_ranges=hp_ranges1,
            candidate_evaluations=_candidate_evaluations(120, hp_ranges1),
            failed_candidates=[],
            pending_evaluations=[]),
    }


def _map_candidate_evaluations(lst, hp_ranges: HyperparameterRanges):
    return [CandidateEvaluation(hp_ranges.tuple_to_config(x), y)
            for x, y in lst]


def _map_pending_evaluations(lst, hp_ranges: HyperparameterRanges):
    return [PendingEvaluation(hp_ranges.tuple_to_config(x)) for x in lst]


@pytest.mark.parametrize('candidate_evaluations,failed_candidates,pending_candidates,expected', [
    ([], [], [], set()),
    ([((123, 'a'), 9.87)], [], [], {(123, 'a')}),
    ([], [(123, 'a')], [], {(123, 'a')}),
    ([], [], [(123, 'a')], {(123, 'a')}),
    ([((1, 'a'), 9.87)], [(2, 'b')], [(3, 'c')],
     {(1, 'a'), (2, 'b'), (3, 'c')})
])
def test_compute_blacklisted_candidates(
        hp_ranges: HyperparameterRanges,
        candidate_evaluations: List[Tuple],
        failed_candidates: List[Tuple],
        pending_candidates: List[Tuple],
        expected: Set[Tuple]):
    state = TuningJobState(
        hp_ranges,
        candidate_evaluations=_map_candidate_evaluations(
            candidate_evaluations, hp_ranges),
        failed_candidates=tuples_to_configs(failed_candidates, hp_ranges),
        pending_evaluations=_map_pending_evaluations(
            pending_candidates, hp_ranges))
    actual = ExclusionList(state)
    assert set(expected) == actual.excl_set


def _assert_no_duplicates(
        candidates: List[Configuration], hp_ranges: HyperparameterRanges):
    cands_tpl = [hp_ranges.config_to_tuple(x) for x in candidates]
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
