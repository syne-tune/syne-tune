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
from typing import Optional, List, Tuple
from operator import itemgetter
import numpy as np
from dataclasses import dataclass


@dataclass
class SlotInRung:
    """
    Used to communicate slot positions and content for them

    """
    rung_index: int  # 0 is lowest rung
    level: int  # Resource level of rung
    slot_index: int  # index of slot in rung
    trial_id: Optional[int]  # None as long as no trial_id assigned
    metric_val: Optional[float]  # Metric value (None if not yet occupied)


class SynchronousHyperbandBracket(object):
    """
    Represents a bracket in synchronous Hyperband.

    A bracket consists of a list of rungs. Each rung consists of a number of
    slots and a resource level (called rung level). The larger the rung level,
    the smaller the number of slots.

    A slot is occupied (by a metric value), free, or pending. A pending slot
    has already been returned by `next_free_slot`. Slots
    in the lowest rung (smallest rung level, largest size) are filled first.
    When a rung is fully occupied, slots for the next rung are assigned with
    the trial_id's having the best metric values. At any point in time, only
    slots in the lowest not fully occupied rung can be filled.

    """
    def __init__(self, rungs: List[Tuple[int, int]], mode: str):
        """
        :param rungs: List of `(rung_size, level)`, where `level` is rung
            (resource) level, `rung_size` is rung size (number of slots).
            All entries must be positive int's. The list must be increasing
            in the first and decreasing in the second component
        :param mode: Criterion is minimized ('min') or maximized ('max')
        """
        self.assert_check_rungs(rungs)
        assert mode in {'min', 'max'}
        # Represents rung levels by (rung, level), where rung is a list of
        # (trial_id, metric_val) tuples for rungs <= self._current_rung.
        # For rungs > self._current_rung, the tuple is (rung_size, level).
        size, level = rungs[0]
        self._rungs = [([(None, None)] * size, level)] + rungs[1:]
        self._mode = mode
        self.current_rung = 0
        self._first_free_pos = 0

    @staticmethod
    def _is_increasing(lst: List[int]) -> bool:
        return all(x < y for x, y in zip(lst, lst[1:]))

    @staticmethod
    def _is_positive_integer(lst: List[int]) -> bool:
        return all(x == int(x) and x >= 1 for x in lst)

    @staticmethod
    def assert_check_rungs(rungs: List[Tuple[int, int]]):
        assert len(rungs) > 0, "There must be at least one rung"
        sizes, levels = zip(*rungs)
        assert SynchronousHyperbandBracket._is_positive_integer(levels), \
            f"Rung levels {levels} are not positive integers"
        assert SynchronousHyperbandBracket._is_increasing(levels), \
            f"Rung levels {levels} are not increasing"
        assert SynchronousHyperbandBracket._is_positive_integer(sizes), \
            f"Rung sizes {sizes} are not positive integers"
        assert SynchronousHyperbandBracket._is_increasing([-x for x in sizes]), \
            f"Rung sizes {sizes} are not decreasing"

    def is_bracket_complete(self) -> bool:
        return self.current_rung >= self.num_rungs

    @property
    def num_rungs(self) -> int:
        return len(self._rungs)

    @staticmethod
    def _trial_ids(rung: List[Tuple[int, float]]) -> List[int]:
        return [x[0] for x in rung]

    def num_pending_slots(self) -> int:
        """
        :return: Number of pending slots (have been returned by
            `next_free_slot`, but not yet occupied
        """
        if self.is_bracket_complete():
            return 0
        rung, _ = self._rungs[self.current_rung]
        return sum(x[1] is None for x in rung[:self._first_free_pos])

    def next_free_slot(self) -> Optional[SlotInRung]:
        if self.is_bracket_complete():
            return None
        rung, milestone = self._rungs[self.current_rung]
        pos = self._first_free_pos
        if pos >= len(rung):
            return None
        trial_id = rung[pos][0]
        self._first_free_pos += 1
        return SlotInRung(
            rung_index=self.current_rung,
            level=milestone,
            slot_index=pos,
            trial_id=trial_id,
            metric_val=None)

    def on_result(self, result: SlotInRung) -> bool:
        """
        Provides result for slot previously requested by `next_free_slot`.
        Here, `result.metric` is written to the slot in order to make it
        occupied. Also, `result.trial_id` is written there if the current
        value is None (happens in the lowest rung).

        :param result: See above
        :return: Has the rung been completely occupied with this result?
        """
        assert result.rung_index == self.current_rung, \
            f"Only accept result for rung index {self.current_rung}:\n" +\
            str(result)
        pos = result.slot_index
        assert 0 <= pos < self._first_free_pos, \
            f"slot_index must be in [0, {self._first_free_pos}):\n" +\
            str(result)
        rung, milestone = self._rungs[self.current_rung]
        assert result.level == milestone, (result, milestone)
        trial_id, metric_val = rung[pos]
        if trial_id is not None:
            assert result.trial_id == trial_id, (result, trial_id)
        assert metric_val is None, \
            f"Slot at {pos} already has metric_val = {metric_val}:\n" +\
            str(result)
        assert result.metric_val is not None, \
            "result.metric_val is missing:\n" + str(result)
        rung[pos] = (result.trial_id, result.metric_val)
        return self._promote_trials_if_rung_complete()

    def _promote_trials_if_rung_complete(self) -> bool:
        rung, _ = self._rungs[self.current_rung]
        is_complete = self._first_free_pos >= len(rung) \
                      and self.num_pending_slots() == 0
        if is_complete:
            self.current_rung += 1
            self._first_free_pos = 0
            if not self.is_bracket_complete():
                pos = self.current_rung
                new_len, milestone = self._rungs[pos]
                # Failed trials insert NaN's
                rung_valid = [x for x in rung if not np.isnan(x[1])]
                num_valid = len(rung_valid)
                if num_valid >= new_len:
                    top_list = sorted(rung_valid, key=itemgetter(1),
                                      reverse=self._mode == 'max')[:new_len]
                else:
                    # Not enough valid entries to fill the new rung (this is
                    # very unlikely to happen). In this case, some failed trials
                    # are still promoted to the next rung.
                    rung_invalid = [x for x in rung if np.isnan(x[1])]
                    top_list = rung_valid + rung_invalid[:(new_len - num_valid)]
                # Set metric_val entries to None, since this distinguishes
                # between a pending and occupied slot
                top_list = [(x[0], None) for x in top_list]
                self._rungs[pos] = (top_list, milestone)
        return is_complete
