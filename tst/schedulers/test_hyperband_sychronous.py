from typing import List, Tuple, Optional
import numpy as np
from collections import Counter
import pytest

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import (
    SynchronousHyperbandBracket,
    SlotInRung,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager import (
    SynchronousHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import (
    SynchronousHyperbandRungSystem,
)


def _trial_ids(lst):
    return [x[0] for x in lst]


def _ask_for_slots(
    bracket: SynchronousHyperbandBracket,
    rung_index: int,
    level: int,
    slot_index: int,
    trial_ids: list,
) -> (List[SlotInRung], int):
    slots = []
    for trial_id in trial_ids:
        slot_in_rung = bracket.next_free_slot()
        assert slot_in_rung is not None
        should_be = SlotInRung(
            rung_index=rung_index,
            level=level,
            slot_index=slot_index,
            trial_id=trial_id,
            metric_val=None,
        )
        assert slot_in_rung == should_be, (slot_in_rung, should_be)
        slots.append(slot_in_rung)
        slot_index += 1
    return slots, slot_index


def _send_results(
    bracket: SynchronousHyperbandBracket,
    slots: List[SlotInRung],
    all_results: List[Tuple[int, float]],
    trials_not_promoted: Optional[List[int]],
):
    trials_n_p = None
    for slot_in_rung in slots:
        trial_id, metric_val = all_results[slot_in_rung.slot_index]
        result = SlotInRung(
            rung_index=slot_in_rung.rung_index,
            level=slot_in_rung.level,
            slot_index=slot_in_rung.slot_index,
            trial_id=trial_id,
            metric_val=metric_val,
        )
        trials_n_p = bracket.on_result(result)
    if trials_not_promoted is None:
        assert trials_n_p is None, trials_n_p
    else:
        assert trials_n_p is not None
        assert set(trials_not_promoted) == set(trials_n_p)


def test_hyperband_bracket():
    rungs = [(9, 1), (4, 3), (1, 9)]
    results = [
        [
            (0, 3.0),
            (1, 5.0),
            (2, 1.0),
            (3, 4.0),
            (4, 9.0),
            (5, 6.0),
            (6, 2.0),
            (7, 7.0),
            (8, 8.0),
        ],
        [(2, 3.1), (6, 3.0), (0, 2.9), (3, 3.0)],
        [(0, 1.0)],
    ]
    trials_not_promoted = [[1, 4, 5, 7, 8], [2, 3, 6], None]
    bracket = SynchronousHyperbandBracket(rungs, mode="min")

    # Rung index 0
    rung_index = 0
    level = rungs[rung_index][1]
    slot_index = 0
    # Ask for some and return before asking for more
    num_jobs = 3
    slots, slot_index = _ask_for_slots(
        bracket, rung_index, level, slot_index, trial_ids=[None] * num_jobs
    )
    assert bracket.num_pending_slots() == num_jobs
    all_results = results[rung_index]
    _send_results(
        bracket=bracket,
        slots=slots,
        all_results=all_results,
        trials_not_promoted=None,
    )
    assert bracket.num_pending_slots() == 0
    # Ask for some, but do not return all for now
    num_jobs = 3
    slots_remaining = []
    for i in range(2):
        slots, slot_index = _ask_for_slots(
            bracket, rung_index, level, slot_index, trial_ids=[None] * num_jobs
        )
        assert bracket.num_pending_slots() == num_jobs + i
        slots_remaining.append(slots[0])
        slots = slots[1:]
        _send_results(
            bracket=bracket,
            slots=slots,
            all_results=all_results,
            trials_not_promoted=None,
        )
        assert bracket.num_pending_slots() == i + 1
    # At this point, there are no free slots, but some are pending
    num_remaining = len(slots_remaining)
    for i, slot in enumerate(slots_remaining):
        assert bracket.next_free_slot() is None
        _send_results(
            bracket=bracket,
            slots=[slot],
            all_results=all_results,
            trials_not_promoted=trials_not_promoted[rung_index]
            if i == num_remaining - 1
            else None,
        )
    # The first rung must be fully occupied now
    assert bracket.num_pending_slots() == 0

    # Other rungs
    for rung_index, all_results in enumerate(results[1:], start=1):
        num_jobs, level = rungs[rung_index]
        slot_index = 0
        trial_ids = [x[0] for x in all_results]
        slots, slot_index = _ask_for_slots(
            bracket, rung_index, level, slot_index, trial_ids=trial_ids
        )
        assert bracket.num_pending_slots() == num_jobs
        assert bracket.next_free_slot() is None
        _send_results(
            bracket=bracket,
            slots=slots,
            all_results=all_results,
            trials_not_promoted=trials_not_promoted[rung_index],
        )
        assert bracket.num_pending_slots() == 0
    # Now, the bracket must be complete
    assert bracket.is_bracket_complete()


def _send_result(
    bracket_manager: SynchronousHyperbandBracketManager,
    slots: List[Tuple[int, SlotInRung]],
    next_trial_id: int,
    random_state: np.random.RandomState,
) -> int:
    bracket_id, slot_in_rung = slots.pop(0)
    if slot_in_rung.trial_id is None:
        slot_in_rung.trial_id = next_trial_id
        next_trial_id += 1
    slot_in_rung.metric_val = random_state.random()
    bracket_manager.on_result((bracket_id, slot_in_rung))
    return next_trial_id


# Runs Hyperband for some number of iterations, checking that no assertions
# are raised
@pytest.mark.timeout(10)
def test_hyperband_bracket_manager_running():
    random_seed = 31415927
    random_state = np.random.RandomState(random_seed)

    bracket_rungs = SynchronousHyperbandRungSystem.geometric(
        min_resource=2, max_resource=200, reduction_factor=3, num_brackets=6
    )
    bracket_manager = SynchronousHyperbandBracketManager(bracket_rungs, mode="min")
    num_jobs = 4
    num_return = 3
    num_steps = 5000
    next_trial_id = 0
    pending_slots = []
    for step in range(num_steps):
        for _ in range(num_jobs):
            pending_slots.append(bracket_manager.next_job())
        # Report results for some, but not all
        for _ in range(num_return):
            next_trial_id = _send_result(
                bracket_manager, pending_slots, next_trial_id, random_state
            )
        # Test whether number of pending are correct
        histogram = Counter([x[0] for x in pending_slots])
        for bracket_id, num_pending in histogram.items():
            assert (
                bracket_manager._brackets[bracket_id].num_pending_slots() == num_pending
            )
        if len(pending_slots) >= 200:
            # Clear all pending slots in random ordering
            for pos in random_state.permutation(len(pending_slots)):
                next_trial_id = _send_result(
                    bracket_manager, [pending_slots[pos]], next_trial_id, random_state
                )
            pending_slots = []
            # Nothing should be pending anymore
            for bracket_id in range(
                bracket_manager._primary_bracket_id, bracket_manager._next_bracket_id
            ):
                assert bracket_manager._brackets[bracket_id].num_pending_slots() == 0
