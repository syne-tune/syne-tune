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
from typing import Tuple
import copy

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import (
    SynchronousHyperbandBracket,
    SlotInRung,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import (
    RungSystemsPerBracket,
)


class SynchronousHyperbandBracketManager:
    """
    Maintains all brackets, relays requests for another job and report of
    result to one of the brackets.

    Each bracket contains a number of rungs, the largest one `max_num_rungs`.
    A bracket with k rungs has offset `max_num_rungs - k`. Hyperband cycles
    through brackets with offset 0, ..., `num_brackets - 1`, where
    `num_brackets <= max_num_rungs`.

    At any given time, one bracket is primary, all other active brackets are
    secondary. Jobs are preferentially assigned to the primary bracket, but
    if its current rung has no free slots (all are pending), secondary
    brackets are considered.

    Each bracket has a bracket_id (nonnegative int), which is used as key for
    the dicts in `next_jobs`, `on_results`. The primary bracket always has
    the lowest id of all active ones. For job assignment, we iterate over
    active brackets starting from the primary, and assign the job to the
    first bracket which has a free slot. If none of the active brackets have
    a free slot, a new bracket is created.

    """

    def __init__(self, bracket_rungs: RungSystemsPerBracket, mode: str):
        """
        :param bracket_rungs: Rungs for successive brackets, from largest to
            smallest
        :param mode: Criterion is minimized ('min') or maximized ('max')
        """
        self.num_bracket_offsets = len(bracket_rungs)
        assert self.num_bracket_offsets > 0
        assert mode in {"min", "max"}
        self.mode = mode
        self.max_num_rungs = len(bracket_rungs[0])
        for offset, rungs in enumerate(bracket_rungs):
            assert len(rungs) == self.max_num_rungs - offset, (
                f"bracket_rungs[{offset}] has size {len(rungs)}, should "
                + f"have size {self.max_num_rungs - offset}"
            )
            SynchronousHyperbandBracket.assert_check_rungs(rungs)
        self._bracket_rungs = copy.deepcopy(bracket_rungs)
        # List of all brackets. We do not delete brackets which are
        # complete, but just keep them for a record
        self._brackets = []
        # Maps bracket_id to offset
        self._bracket_id_to_offset = []
        # Maps (offset, level), level a rung level in the bracket, to
        # the previous rung level (or 0)
        self._level_to_prev_level = dict()
        for offset, rungs in enumerate(bracket_rungs):
            _, levels = zip(*rungs)
            levels = (0,) + levels
            self._level_to_prev_level.update(
                ((offset, lv), plv) for (lv, plv) in zip(levels[1:], levels[:-1])
            )
        # Create primary bracket
        self._primary_bracket_id = self._create_new_bracket()

    @property
    def bracket_rungs(self) -> RungSystemsPerBracket:
        return self._bracket_rungs

    @property
    def _next_bracket_id(self) -> int:
        return len(self._brackets)

    def level_to_prev_level(self, bracket_id: int, level: int) -> int:
        """
        :param bracket_id:
        :param level: Level in bracket
        :return: Previous level; or 0
        """
        offset = self._bracket_id_to_offset[bracket_id]
        return self._level_to_prev_level[(offset, level)]

    def _create_new_bracket(self) -> int:
        # Sanity check:
        assert len(self._brackets) == len(self._bracket_id_to_offset)
        bracket_id = self._next_bracket_id
        offset = bracket_id % self.num_bracket_offsets
        self._bracket_id_to_offset.append(offset)
        self._brackets.append(
            SynchronousHyperbandBracket(self._bracket_rungs[offset], self.mode)
        )
        return bracket_id

    def next_job(self) -> Tuple[int, SlotInRung]:
        """
        Called by scheduler to request a new job. Jobs are preferentially
        assigned to the primary bracket, which has the lowest id among all
        active brackets. If the primary bracket does not accept jobs (because
        all remaining slots are already pending), further active brackets are
        polled. If none of the active brackets accept jobs, a new bracket is
        created.

        The job description returned is (bracket_id, slot_in_rung), where
        `slot_in_rung` is :class:`SlotInRung`, containing the info of what
        is to be done (`trial_id`, `level` fields). It is this entry which
        has to be returned in 'on_result`, which the `metric_val` field set.
        If the job returned here has `trial_id == None`, it comes from the
        lowest rung of its bracket, and the `trial_id` has to be set as well
        when returning the record in `on_result`.

        :return: Tuple (bracket_id, slot_in_rung)
        """
        # Try to assign job to active bracket. There must be at least one,
        # the primary one
        bracket_ids = range(self._primary_bracket_id, self._next_bracket_id)
        for bracket_id in bracket_ids:
            slot_in_rung = self._brackets[bracket_id].next_free_slot()
            if slot_in_rung is not None:
                return bracket_id, slot_in_rung
        # None of the existing brackets accept jobs. Create a new one
        bracket_id = self._create_new_bracket()
        slot_in_rung = self._brackets[bracket_id].next_free_slot()
        assert slot_in_rung is not None, "Newly created bracket has to have a free slot"
        return bracket_id, slot_in_rung

    def on_result(self, result: Tuple[int, SlotInRung]):
        """
        Called by scheduler to provide result for previously requested job.
        See `next_job`.

        :param result: Tuple (bracket_id, slot_in_rung)
        """
        bracket_id, slot_in_rung = result
        assert self._primary_bracket_id <= bracket_id < self._next_bracket_id, (
            f"Invalid bracket_id = {bracket_id}, must be in "
            + f"[{self._primary_bracket_id}, {self._next_bracket_id})"
        )
        bracket = self._brackets[bracket_id]
        bracket.on_result(slot_in_rung)
        for_primary = bracket_id == self._primary_bracket_id
        if for_primary:
            # Primary bracket is complete: Move to next one. While very
            # unlikely, brackets after the primary one could be complete
            # as well
            last_bracket = self._next_bracket_id - 1
            while (
                bracket.is_bracket_complete()
                and self._primary_bracket_id < last_bracket
            ):
                self._primary_bracket_id += 1
                bracket = self._brackets[self._primary_bracket_id]
            # May have to create a new bracket
            if bracket.is_bracket_complete():
                self._primary_bracket_id = self._create_new_bracket()
