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
from typing import List, Tuple, Optional

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager import (
    SynchronousHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.dehb_bracket import (
    DifferentialEvolutionHyperbandBracket,
)


class DifferentialEvolutionHyperbandBracketManager(SynchronousHyperbandBracketManager):
    """
    Special case of :class:`SynchronousHyperbandBracketManager` to manage DEHB
    brackets (type :class:`DifferentialEvolutionHyperbandBracket`).

    In DEHB, the list of brackets is determined by the first one and the number
    of brackets. Also, later brackets have less total budget, because the size
    of a rung is determined by its level, independent of the bracket. This is
    different to what is done in synchronous Hyperband, where the rungs of
    later brackets have larger sizes, so the total budget of each bracket is
    the same.

    We also need additional methods to access trial_id's in specific rungs, as
    well as entries of the top lists for completed rungs. This is because DEHB
    controls the creation of new configurations at higher rungs, while
    synchronous Hyperband relies on automatic promotion from lower rungs.
    """

    def __init__(
        self,
        rungs_first_bracket: List[Tuple[int, int]],
        mode: str,
        num_brackets_per_iteration: Optional[int] = None,
    ):
        max_num_offsets = len(rungs_first_bracket)
        assert max_num_offsets > 0
        if num_brackets_per_iteration is None:
            num_brackets_per_iteration = max_num_offsets
        else:
            assert 1 <= num_brackets_per_iteration <= max_num_offsets, (
                f"num_brackets_per_iteration = {num_brackets_per_iteration}"
                + f", must be in [1, {max_num_offsets}]"
            )
        # All brackets are determined by the first one in DEHB
        bracket_rungs = [
            rungs_first_bracket[offset:] for offset in range(num_brackets_per_iteration)
        ]
        super().__init__(bracket_rungs, mode)
        # Maps (bracket_id, rung_index) to top list of previous rung, as
        # returned by
        # `DifferentialEvolutionHyperbandBracket.top_list_for_previous_rung`
        # when the current rung is `rung_index` in that bracket. We cache
        # these, so we don't have to repeat sorting many times
        self._top_list_of_previous_rung_cache = dict()
        # Maps (offset, level) to (bracket_delta, rung_index) in order to
        # determine the parent rung of a rung in a bracket with offset and
        # level (the parent rung has the same level).
        self._parent_rung = self._set_parent_rung()

    def _set_parent_rung(self):
        parent_rung = dict()
        for offset, rungs in enumerate(self._bracket_rungs):
            if offset > 0:
                # For bracket with offset > 0, the parent rung is in the
                # bracket just to the left
                bracket_delta = 1
                for rung_index, (_, level) in enumerate(rungs):
                    parent_rung[(offset, level)] = (
                        bracket_delta,
                        rung_index + 1,
                    )
            else:
                # For bracket with offset 0, the parent rung is the base
                # rung in a bracket to the left
                for rung_index, (_, level) in enumerate(rungs):
                    parent_rung[(offset, level)] = (
                        self.num_bracket_offsets - rung_index,
                        0,
                    )
        return parent_rung

    def _create_new_bracket(self) -> int:
        # Sanity check:
        assert len(self._brackets) == len(self._bracket_id_to_offset)
        bracket_id = self._next_bracket_id
        offset = bracket_id % self.num_bracket_offsets
        self._bracket_id_to_offset.append(offset)
        rungs = self._bracket_rungs[offset]
        self._brackets.append(
            DifferentialEvolutionHyperbandBracket(rungs=rungs, mode=self.mode)
        )
        return bracket_id

    def size_of_current_rung(self, bracket_id: int) -> int:
        return self._brackets[bracket_id].size_of_current_rung()

    def trial_id_from_parent_slot(
        self, bracket_id: int, level: int, slot_index: int
    ) -> Optional[int]:
        """
        The parent slot has the same slot index and rung level in the
        largest bracket `< bracket_id` with a trial_id not None. If no
        such slot exists, None is returned.
        For a cross-over or selection operation, the target is chosen
        from the parent slot.
        """
        trial_id = None
        while trial_id is None and bracket_id > 0:
            bracket_delta, rung_index = self._parent_rung[
                (self._bracket_id_to_offset[bracket_id], level)
            ]
            bracket_id = bracket_id - bracket_delta
            trial_id = self._brackets[bracket_id].trial_id_for_slot(
                rung_index=rung_index, slot_index=slot_index
            )
        return trial_id

    def top_of_previous_rung(self, bracket_id: int, pos: int) -> int:
        """
        For the current rung in bracket `bracket_id`, consider the slots of
        the previous rung (below) in sorted order. We return the trial_id of
        position `pos` (so for `pos=0`, the best entry).
        """
        bracket = self._brackets[bracket_id]
        rung_index = bracket.current_rung
        key = (bracket_id, rung_index)
        top_list = self._top_list_of_previous_rung_cache.get(key)
        if top_list is None:
            top_list = bracket.top_list_for_previous_rung()
            self._top_list_of_previous_rung_cache[key] = top_list
        return top_list[pos]
