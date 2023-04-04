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
import copy
from numpy.random import RandomState
from operator import itemgetter

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.utils.successive_halving import (
    successive_halving_rung_levels,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    StateForModelConverter,
)
from syne_tune.util import is_positive_integer


ObservedData = List[Tuple[int, int, float]]


def _extract_observations(
    trials_evaluations: List[TrialEvaluations],
) -> ObservedData:
    """
    Maps ``trials_evaluations`` to list of tuples :math:`(i, r, y_{i r})`, where
    :math:`y_{i r}` is the observed value for trial ID :math:`i` at resource
    level :math:`r`.

    :param trials_evaluations: See above
    :return: See above
    """
    all_data = []
    for trial_eval in trials_evaluations:
        trial_id = int(trial_eval.trial_id)
        all_data.extend(
            [
                (trial_id, int(r), y)
                for r, y in trial_eval.metrics[INTERNAL_METRIC_NAME].items()
            ]
        )
    return all_data


def _create_trials_evaluations(data: ObservedData) -> List[TrialEvaluations]:
    """Inverse of :func:`_extract_observations`

    :param data: List of tuples
    :return: Resulting ``trials_evaluations``
    """
    trials_evaluations = dict()
    for (trial_id, level, fval) in data:
        trial_id = str(trial_id)
        if trial_id in trials_evaluations:
            trials_evaluations[trial_id][str(level)] = fval
        else:
            trials_evaluations[trial_id] = {str(level): fval}
    return [
        TrialEvaluations(trial_id=trial_id, metrics={INTERNAL_METRIC_NAME: metrics})
        for trial_id, metrics in trials_evaluations.items()
    ]


def cap_size_tuning_job_state(
    state: TuningJobState,
    max_size: int,
    random_state: Optional[RandomState] = None,
) -> TuningJobState:
    """
    Returns state which is identical to ``state``, except that the
    ``trials_evaluations`` are replaced by a subset so the total number of
    metric values is ``<= max_size``. Filtering is done by preserving data
    from trials which have observations at the higher resource levels. For
    some trials, we may remove values at low resources, but keep values at
    higher ones, in order to meet the ``max_size`` constraint.

    :param state: Original state to filter down
    :param max_size: Maximum number of observed metric values in new state
    :param random_state: Used for random sampling. Defaults to ``numpy.random``.
    :return: New state meeting the ``max_size`` constraint. This is a copy of
        ``state`` even if this meets the constraint already.
    """
    total_size = state.num_observed_cases()
    if total_size <= max_size:
        trials_evaluations = copy.deepcopy(state.trials_evaluations)
    else:
        # Need to do subsampling. This is done bottom-up (adding entries
        # to ``new_data``) instead of top-down by filtering
        remaining_data = _extract_observations(state.trials_evaluations)
        new_data = []
        rung_levels = sorted(list({r for _, r, _ in remaining_data}))
        for level in reversed(rung_levels):
            # Trials with data at level ``level``
            trial_ids = set(i for i, r, _ in remaining_data if r == level)
            # Data of these trials should be retained preferentially
            next_data = [x for x in remaining_data if x[0] in trial_ids]
            remaining_data = [x for x in remaining_data if x[0] not in trial_ids]
            next_size = max_size - len(new_data)  # Remaining gap
            if len(next_data) > next_size:
                # Subsample ``next_data`` to size ``next_size``, which will
                # fill the remaining gap
                for low_level in (r for r in rung_levels if r <= level):
                    if low_level < level:
                        sample_data = [x for x in next_data if x[1] == low_level]
                        next_data = [x for x in next_data if x[1] > low_level]
                    else:
                        sample_data = next_data
                        next_data = []
                    # If ``next_data`` is still larger than ``next_size``, drop
                    # ``sample_data`` and go to next ``low_level``
                    sample_size = next_size - len(next_data)
                    if sample_size > 0:
                        index = random_state.choice(
                            len(sample_data), size=sample_size, replace=False
                        )
                        next_data.extend(sample_data[pos] for pos in index)
                        break
                assert len(next_data) == next_size  # Sanity check
            new_data.extend(next_data)
            if len(new_data) == max_size:
                break
        assert len(new_data) == max_size  # Sanity check
        trials_evaluations = _create_trials_evaluations(new_data)
    return TuningJobState(
        hp_ranges=state.hp_ranges,
        config_for_trial=state.config_for_trial.copy(),
        trials_evaluations=trials_evaluations,
        failed_trials=state.failed_trials.copy(),
        pending_evaluations=state.pending_evaluations.copy(),
    )


class SubsampleMultiFidelityStateConverter(StateForModelConverter):
    """
    Converts state by (possibly) down sampling the observation so that their
    total number is ``<= max_size``. This is done in a way that trials with
    observations in higher rung levels are retained (with all their data),
    so observations are preferentially removed at low levels, and from trials
    which do not have observations higher up.

    This state converter makes sense if observed data is only used at geometrically
    spaced rung levels, so the number of observations per trial remains small. If
    a trial runs up on the order of ``max_resource_level`` observations, it does
    not work, because it ends up retaining densely sampled observations from very
    few trials. Use :class:`SubsampleMFDenseDataStateConverter` in such a case.
    """

    def __init__(self, max_size: int, random_state: Optional[RandomState] = None):
        self.max_size = int(max_size)
        assert self.max_size >= 1
        self._random_state = random_state

    def __call__(self, state: TuningJobState) -> TuningJobState:
        assert (
            self._random_state is not None
        ), "Call set_random_state before first usage"
        return cap_size_tuning_job_state(
            state=state, max_size=self.max_size, random_state=self._random_state
        )

    def set_random_state(self, random_state: RandomState):
        self._random_state = random_state


def _sparsify_at_most_one_per_trial_and_group(
    data: ObservedData, max_size: int, rung_levels: List[int]
) -> ObservedData:
    r"""
    Define groups :math:`G_j = \{r_j + 1,\dots, r_j\}`, where math:`[r_j]` is
    ``rung_levels``. We filter ``data`` so that for every trial ID :math:`i` in
    ``data``, only the entry with the largest resource :math:`r` in each group
    :math:`G_j` is retained. Filtering is done w.r.t. descending :math:`r`, and
    is stopped once the size ``max_size`` is reached.
    """
    return_size = len(data)
    if return_size <= max_size:
        return data
    # Map r -> group
    padded_rung_levels = [-1] + rung_levels
    resource_to_group = dict()
    for g, (r_low, r_high) in enumerate(
        zip(padded_rung_levels[:-1], padded_rung_levels[1:])
    ):
        for r in range(r_low + 1, r_high + 1):
            resource_to_group[r] = g
    # Filter in order (r, i) descending, so for the same r, trial IDs are
    # considered largest first
    trial_group_covered = set()
    new_data = []
    sorted_data = sorted(data, key=itemgetter(1, 0), reverse=True)
    for pos, elem in enumerate(sorted_data):
        trial_id, resource, _ = elem
        trial_group = (trial_id, resource_to_group[resource])
        if trial_group not in trial_group_covered:
            new_data.append(elem)
            trial_group_covered.add(trial_group)
        else:
            return_size -= 1
            if return_size == max_size:
                break
    new_data.extend(sorted_data[(pos + 1) :])
    len_new_data = len(new_data)
    assert return_size == len_new_data, (return_size, len_new_data)
    return new_data


def sparsify_tuning_job_state(
    state: TuningJobState, max_size: int, grace_period: int, reduction_factor: float
) -> TuningJobState:
    r"""
    Does the first step of state conversion in
    :class:`SubsampleMFDenseDataStateConverter`, in that dense observations are
    sparsified w.r.t. a geometrically spaced rung level system.

    :param state: Original state to filter down
    :param max_size: Maximum number of observed metric values in new state
    :param grace_period: Minimum resource level :math:`r_{min}`
    :param reduction_factor: Reduction factor :math:`\eta`
    :return: New state which either meets the ``max_size`` constraint, or is
        maximally sparsified
    """
    total_size = state.num_observed_cases()
    if total_size <= max_size:
        trials_evaluations = copy.deepcopy(state.trials_evaluations)
    else:
        # Need to do sparsification
        data = _extract_observations(state.trials_evaluations)
        max_t = max(r for _, r, _ in data)
        assert max_t >= grace_period  # Sanity check
        if max_t == grace_period:
            trials_evaluations = copy.deepcopy(state.trials_evaluations)
        else:
            rung_levels = successive_halving_rung_levels(
                rung_levels=None,
                grace_period=grace_period,
                reduction_factor=reduction_factor,
                rung_increment=None,
                max_t=max_t,
            ) + [max_t]
            new_data = _sparsify_at_most_one_per_trial_and_group(
                data, max_size, rung_levels
            )
            trials_evaluations = _create_trials_evaluations(new_data)
    return TuningJobState(
        hp_ranges=state.hp_ranges,
        config_for_trial=state.config_for_trial.copy(),
        trials_evaluations=trials_evaluations,
        failed_trials=state.failed_trials.copy(),
        pending_evaluations=state.pending_evaluations.copy(),
    )


class SubsampleMFDenseDataStateConverter(SubsampleMultiFidelityStateConverter):
    r"""
    Variant of :class:`SubsampleMultiFidelityStateConverter`, which has the same
    goal, but does subsampling in a different way. The current default for most
    GP-based multi-fidelity algorithms (e.g., MOBSTER, Hyper-Tune) is to use
    observations only at geometrically spaced rung levels (such as 1, 3, 9, ...),
    and :class:`SubsampleMultiFidelityStateConverter` makes sense.

    But for some (e.g., DyHPO), observations are recorded at all (or linearly
    spaced) resource levels, so there is much more data for trials which progressed
    further. Here, we do the state conversion in two steps, always stopping the
    process once the target size ``max_size`` is reached. We assume a geometric
    rung level spacing, given by ``grace_period`` and ``reduction_factor``, only
    for the purpose of state conversion. In the first step, we sparsify the
    observations. If each rung level :math:`r_k`` defines a bucket
    :math:`B_k = r_{k-1} + 1, \dots, r_k`, each trial should have at most one
    observation in each bucket. Sparsification is done top down. If the result of
    this first step is still larger than ``max_size``, we continue with subsampling
    as in :class:`SubsampleMultiFidelityStateConverter`.
    """

    def __init__(
        self,
        max_size: int,
        grace_period: Optional[int] = None,
        reduction_factor: Optional[float] = None,
        random_state: Optional[RandomState] = None,
    ):
        super().__init__(max_size, random_state)
        if grace_period is None:
            grace_period = 1
        else:
            assert is_positive_integer([grace_period])
        if reduction_factor is None:
            reduction_factor = 3
        else:
            assert reduction_factor >= 2
        self._grace_period = grace_period
        self._reduction_factor = reduction_factor

    def __call__(self, state: TuningJobState) -> TuningJobState:
        assert (
            self._random_state is not None
        ), "Call set_random_state before first usage"
        # First step: Reduce number of observation by sparsification. This does
        # not involve random sampling
        new_state = sparsify_tuning_job_state(
            state=state,
            max_size=self.max_size,
            grace_period=self._grace_period,
            reduction_factor=self._reduction_factor,
        )
        # Second step: If result still has too many observations, call super-class
        # subsampling
        size_new_state = new_state.num_observed_cases()
        if size_new_state > self.max_size:
            new_state = super().__call__(new_state)
        return new_state
