from typing import Optional, List, Tuple
from operator import itemgetter
import numpy as np
from dataclasses import dataclass

from syne_tune.util import is_increasing, is_positive_integer


@dataclass
class SlotInRung:
    """
    Used to communicate slot positions and content for them.
    """

    rung_index: int  # 0 is lowest rung
    level: int  # Resource level of rung
    slot_index: int  # index of slot in rung
    trial_id: Optional[int]  # None as long as no trial_id assigned
    metric_val: Optional[float]  # Metric value (None if not yet occupied)


class SynchronousBracket:
    """
    Base class for a single bracket in synchronous Hyperband algorithms.

    A bracket consists of a list of rungs. Each rung consists of a number of
    slots and a resource level (called rung level). The larger the rung level,
    the smaller the number of slots.

    A slot is occupied (by a metric value), free, or pending. A pending slot
    has already been returned by :meth:`next_free_slot`. Slots
    in the lowest rung (smallest rung level, largest size) are filled first.
    At any point in time, only slots in the lowest not fully occupied rung
    can be filled. If there are no free slots in the current rung, but there
    are pending ones, the bracket is blocked, and another bracket needs to
    be worked on.
    """

    def __init__(self, mode: str):
        assert mode in {"min", "max"}
        self._mode = mode
        self._first_free_pos = 0
        self.current_rung = 0

    @staticmethod
    def assert_check_rungs(rungs: List[Tuple[int, int]]):
        assert len(rungs) > 0, "There must be at least one rung"
        sizes, levels = zip(*rungs)
        assert is_positive_integer(
            levels
        ), f"Rung levels {levels} are not positive integers"
        assert is_increasing(levels), f"Rung levels {levels} are not increasing"
        assert is_positive_integer(
            sizes
        ), f"Rung sizes {sizes} are not positive integers"
        assert is_increasing(
            [-x for x in sizes]
        ), f"Rung sizes {sizes} are not decreasing"

    @property
    def num_rungs(self) -> int:
        raise NotImplementedError

    def is_bracket_complete(self) -> bool:
        return self.current_rung >= self.num_rungs

    def _current_rung_and_level(
        self,
    ) -> (List[Tuple[Optional[int], Optional[float]]], int):
        raise NotImplementedError

    def num_pending_slots(self) -> int:
        """
        :return: Number of pending slots (have been returned by
            ``next_free_slot``, but not yet occupied
        """
        if self.is_bracket_complete():
            return 0
        rung, _ = self._current_rung_and_level()
        return sum(x[1] is None for x in rung[: self._first_free_pos])

    def next_free_slot(self) -> Optional[SlotInRung]:
        if self.is_bracket_complete():
            return None
        rung, milestone = self._current_rung_and_level()
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
            metric_val=None,
        )

    def on_result(self, result: SlotInRung) -> Optional[List[int]]:
        """
        Provides result for slot previously requested by ``next_free_slot``.
        Here, ``result.metric`` is written to the slot in order to make it
        occupied. Also, ``result.trial_id`` is written there.

        We normally return ``None``. But if the result passed completes the
        current rung, this triggers the creation of a child run which consists
        of promoted trials from the current rung. In this case, we return the
        IDs of trials which have not been promoted. This is used in for early
        removal of checkpoints, see
        :meth:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler.trials_checkpoints_can_be_removed`.

        :param result: See above
        :return: See above
        """
        assert (
            result.rung_index == self.current_rung
        ), f"Only accept result for rung index {self.current_rung}:\n" + str(result)
        pos = result.slot_index
        assert (
            0 <= pos < self._first_free_pos
        ), f"slot_index must be in [0, {self._first_free_pos}):\n" + str(result)
        rung, milestone = self._current_rung_and_level()
        assert result.level == milestone, (result, milestone)
        trial_id, metric_val = rung[pos]
        self._assert_on_result_trial_id(result, trial_id)
        assert (
            metric_val is None
        ), f"Slot at {pos} already has metric_val = {metric_val}:\n" + str(result)
        assert result.metric_val is not None, "result.metric_val is missing:\n" + str(
            result
        )
        rung[pos] = (result.trial_id, result.metric_val)
        # Check whether rung is complete. If so, move to next one and trigger
        # promotions (optional)
        is_complete = (
            self._first_free_pos >= len(rung) and self.num_pending_slots() == 0
        )
        trials_not_promoted = None
        if is_complete:
            self.current_rung += 1
            self._first_free_pos = 0
            if not self.is_bracket_complete():
                trials_not_promoted = self._promote_trials_at_rung_complete()
        return trials_not_promoted

    def _assert_on_result_trial_id(self, result: SlotInRung, trial_id: int):
        pass

    def _promote_trials_at_rung_complete(self) -> List[int]:
        """
        This is called when a rung has just been filled in :meth:`on_result`,
        and this is not the final rung of the bracket. It opens a new rung
        by promoting top performing trials from the current one.

        :return: List of trials which are not promoted
        """
        raise NotImplementedError


class SynchronousHyperbandBracket(SynchronousBracket):
    """
    Represents a bracket in standard synchronous Hyperband.

    When a rung is fully occupied, slots for the next rung are assigned with
    the trial_id's having the best metric values. At any point in time, only
    slots in the lowest not fully occupied rung can be filled.
    """

    def __init__(self, rungs: List[Tuple[int, int]], mode: str):
        """
        :param rungs: List of ``(rung_size, level)``, where ``level`` is rung
            (resource) level, ``rung_size`` is rung size (number of slots).
            All entries must be positive int's. The list must be increasing
            in the first and decreasing in the second component
        :param mode: Criterion is minimized ('min') or maximized ('max')
        """
        self.assert_check_rungs(rungs)
        super().__init__(mode)
        # Represents rung levels by (rung, level), where rung is a list of
        # (trial_id, metric_val) tuples for rungs <= self._current_rung.
        # For rungs > self._current_rung, the tuple is (rung_size, level).
        size, level = rungs[0]
        self._rungs = [([(None, None)] * size, level)] + rungs[1:]

    @property
    def num_rungs(self) -> int:
        return len(self._rungs)

    def _current_rung_and_level(
        self,
    ) -> (List[Tuple[Optional[int], Optional[float]]], int):
        return self._rungs[self.current_rung]

    def _assert_on_result_trial_id(self, result: SlotInRung, trial_id: int):
        if trial_id is not None:
            assert result.trial_id == trial_id, (result, trial_id)

    def _promote_trials_at_rung_complete(self) -> List[int]:
        pos = self.current_rung
        new_len, milestone = self._rungs[pos]
        previous_rung, _ = self._rungs[pos - 1]
        top_list, remaining_list = get_top_list(
            rung=previous_rung, new_len=new_len, mode=self._mode
        )
        # Set metric_val entries to None, since this distinguishes
        # between a pending and occupied slot
        top_list = [(trial_id, None) for trial_id in top_list]
        self._rungs[pos] = (top_list, milestone)
        return remaining_list


def get_top_list(
    rung: List[Tuple[int, float]], new_len: int, mode: str
) -> (List[int], List[int]):
    """
    Returns list of IDs of trials of len ``new_len`` which should be promoted,
    because they performed best. We also return the list of IDs of the remaining
    trials, which are not to be promoted.

    :param rung: Current rung which has just been completed
    :param new_len: Size of new rung
    :param mode: "min" or "max"
    :return: ``(top_list, remaining_list)``
    """
    # Failed trials insert NaN's
    rung_valid = [x for x in rung if not np.isnan(x[1])]
    num_valid = len(rung_valid)
    if num_valid >= new_len:
        top_list = [
            x[0]
            for x in sorted(rung_valid, key=itemgetter(1), reverse=mode == "max")[
                :new_len
            ]
        ]
    else:
        # Not enough valid entries to fill the new rung (this is
        # very unlikely to happen). In this case, some failed trials
        # are still promoted to the next rung.
        invalid_list = [x[0] for x in rung if np.isnan(x[1])]
        top_list = [x[0] for x in rung_valid] + invalid_list[: (new_len - num_valid)]
    top_set = set(top_list)
    remaining_list = [x[0] for x in rung if x[0] not in top_set]
    return top_list, remaining_list
