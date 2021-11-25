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


class SynchronousHyperbandBracket(object):
    """
    Represents a bracket in synchronous Hyperband.

    A bracket consists of a list of rungs. Each rung consists of a number of
    slots and a resource level (called rung level). The larger the rung level,
    the smaller the number of slots.

    A slot is either occupied by a trial_id and metric value, or free. Slots
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

    def next_slots(self) -> Optional[Tuple[List[Optional[int]], int]]:
        """
        Returns tuple `(trials, milestone)`, where `trials` is a list of trial_id's
        to be promoted to resource level `milestone`, in order to complete the
        current rung. If the current rung is the lowest, its trials do not exist
        yet and have to be started from scratch, in which case `trials` is a list
        of `None` values of the required size (to complete the rung).

        Finally, if the bracket is complete, `None` is returned instead

        :return: (trials, milestone) or None
        """
        if self.is_bracket_complete():
            return None
        rung, milestone = self._rungs[self.current_rung]
        pos = self._first_free_pos
        return self._trial_ids(rung[pos:]), milestone

    def on_results(self, results: List[Tuple[int, float]]) -> bool:
        """
        Provides result values for the next free slots. `results` contains
        tuples `(trial_id, metric_val)`, so that the list of trial_id's
        corresponds to the prefix of `trials` returned by `next_slots` (unless
        `trials` consists of `None` values, in which case the trial_id's have
        just been assigned.

        :param results: See above
        :return: Have these results led to the current rung being completely
            filled?
        """
        self._assert_check_results(results)
        rung, _ = self._rungs[self.current_rung]
        num_results = len(results)
        pos = self._first_free_pos
        rung[pos:(pos + num_results)] = results
        self._first_free_pos += num_results
        return self._promote_trials_if_rung_complete()

    def _assert_check_results(self, results: List[Tuple[int, float]]):
        assert not self.is_bracket_complete(), \
            "This bracket is complete, no more results are needed"
        _next_slots, _ = self.next_slots()
        num_results = len(results)
        assert num_results <= len(_next_slots), \
            f"There are only {len(_next_slots)} free slots in the current " +\
            f"bracket, but len(results) = {num_results}"
        if self.current_rung > 0:
            trial_ids_results = self._trial_ids(results)
            assert trial_ids_results == _next_slots[:num_results], \
                f"trial_ids of results ({trial_ids_results}) not the same " +\
                f"as trial_ids of free slots ({_next_slots[:num_results]})"

    def _promote_trials_if_rung_complete(self) -> bool:
        rung, _ = self._rungs[self.current_rung]
        is_complete = self._first_free_pos >= len(rung)
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
                # For the (trial_id, metric_val) entries, only trial_id is
                # valid (while metric_val is arbitrary) for positions
                # >= self._first_free_pos
                self._rungs[pos] = (top_list, milestone)
        return is_complete