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
from typing import Optional, List, Tuple, Dict
import copy

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import \
    SynchronousHyperbandBracket
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system \
    import RungSystemsPerBracket


class SynchronousHyperbandBracketManager(object):
    """
    Maintains all currently active brackets, assigns batches of jobs to
    brackets.

    Each bracket contains a number of rungs, the largest one `max_num_rungs`.
    A bracket with k rungs has offset `max_num_rungs - k`. Hyperband cycles
    through brackets with offset 0, ..., `num_brackets - 1`, where
    `num_brackets <= max_num_rungs`.

    At any given time, one bracket is primary, all other active brackets are
    secondary. Jobs are preferentially assigned to the primary bracket, but
    if its current rung has fewer free slots than jobs to be assigned,
    secondary brackets are considered. There is at most one secondary bracket
    for each offset.

    Each bracket has a bracket_id (nonnegative int), which is used as key for
    the dicts in `next_jobs`, `on_results`.

    """
    def __init__(self, bracket_rungs: RungSystemsPerBracket, mode: str):
        """
        :param bracket_rungs: Rungs for successive brackets, from largest to
            smallest
        :param mode: Criterion is minimized ('min') or maximized ('max')
        """
        self.num_brackets = len(bracket_rungs)
        assert self.num_brackets > 0
        assert mode in {'min', 'max'}
        self.mode = mode
        self.max_num_rungs = len(bracket_rungs[0])
        for offset, rungs in enumerate(bracket_rungs):
            assert len(rungs) == self.max_num_rungs - offset, \
                f"bracket_rungs[{offset}] has size {len(rungs)}, should " +\
                f"have size {self.max_num_rungs - offset}"
            SynchronousHyperbandBracket.assert_check_rungs(rungs)
        self._bracket_rungs = copy.deepcopy(bracket_rungs)
        # Dictionary of all active brackets
        self._brackets = dict()
        # Primary bracket
        self._primary_offset = 0
        self._next_bracket_id = 0
        self._primary_bracket_id = self._create_new_bracket(
            self._primary_offset)
        # Maps offset to secondary bracket_id
        self._offset_to_secondary_bracket_id = [None] * self.num_brackets

    @property
    def bracket_rungs(self) -> RungSystemsPerBracket:
        return self._bracket_rungs

    def _create_new_bracket(self, offset: int) -> int:
        bracket_id = self._next_bracket_id
        self._brackets[bracket_id] = SynchronousHyperbandBracket(
            self._bracket_rungs[offset], self.mode)
        self._next_bracket_id += 1
        return bracket_id

    def _bracket_id_from_offset(self, offset: int):
        """
        Creates new secondary bracket if it does not exist already

        """
        bracket_id = self._offset_to_secondary_bracket_id[offset]
        if bracket_id is None:
            bracket_id = self._create_new_bracket(offset)
            self._offset_to_secondary_bracket_id[offset] = bracket_id
        return bracket_id

    def _assign_jobs(
            self, bracket_id: int, num_jobs: int, result: list) -> int:
        bracket = self._brackets[bracket_id]
        trials, milestone = bracket.next_slots()
        num_assigned = min(num_jobs, len(trials))
        result.append((bracket_id, (trials[:num_assigned], milestone)))
        return num_assigned

    def next_jobs(
            self, num_jobs) -> List[Tuple[int, Tuple[List[Optional[int]],
                                                     int]]]:
        """
        Called by scheduler to request `num_jobs` jobs. Jobs are preferentially
        assigned to the primary bracket, but if this is not possible, a secondary
        bracket is also used (possibly even several ones).

        Returns list of `(bracket_id, (trials, milestone))`, where
        `(trials, milestone)` come from `next_slots` of bracket `bracket_id`.
        There, the sum of `len(trials)` is equal to `num_jobs`.

        :param num_jobs:
        :return:
        """
        result = []
        num_done = self._assign_jobs(
            self._primary_bracket_id, num_jobs, result)
        if num_done < num_jobs:
            # num_jobs - num_done jobs to be assigned to secondary bracket(s)
            primary_bracket = self._brackets[self._primary_bracket_id]
            rung_ind = primary_bracket.current_rung
            # Determine preferred offset for secondary bracket
            if rung_ind == 0:
                secondary_offset = self._primary_offset
            else:
                secondary_offset = min(
                    self._primary_offset + rung_ind, self.num_brackets) - 1
            # First entry in `offsets` is preferred one
            offsets = [x % self.num_brackets for x in range(
                secondary_offset, secondary_offset + self.num_brackets)]
            while num_done < num_jobs and offsets:
                secondary_offset = offsets.pop(0)
                # Creates new bracket for `secondary_offset` if none exists
                bracket_id = self._bracket_id_from_offset(secondary_offset)
                num_extra = self._assign_jobs(
                    bracket_id, num_jobs - num_done, result)
                num_done += num_extra
            assert num_done == num_jobs, \
                f"Could only assign {num_done} of {num_jobs} jobs: {result}"
        return result

    def on_results(
            self, bracket_to_results: Dict[int, List[Tuple[int, float]]]):
        """
        Called by scheduler to provide results for jobs previously
        requested.

        :param bracket_to_results:
        """
        primary_done = False
        reassign_primary = False
        for bracket_id, results in bracket_to_results.items():
            bracket = self._brackets[bracket_id]
            bracket.on_results(results)
            bracket_is_complete = bracket.is_bracket_complete()
            if bracket_id == self._primary_bracket_id:
                primary_done = True
                reassign_primary = bracket_is_complete
            elif bracket_is_complete:
                # Secondary bracket complete (very unlikely to happen!)
                offset = self.max_num_rungs - bracket.num_rungs
                if bracket_id == self._offset_to_secondary_bracket_id[offset]:
                    self._offset_to_secondary_bracket_id[offset] = None
                del self._brackets[bracket_id]
        assert primary_done, \
            f"Results for primary bracket (id {self._primary_bracket_id}) " +\
            "not provided"
        if reassign_primary:
            del self._brackets[self._primary_bracket_id]
            self._primary_offset = \
                (self._primary_offset + 1) % self.num_brackets
            self._primary_bracket_id = self._bracket_id_from_offset(
                self._primary_offset)
            self._offset_to_secondary_bracket_id[self._primary_offset] = None
