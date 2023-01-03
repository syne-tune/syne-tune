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

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    StateForModelConverter,
)


def _extract_observations(
    trials_evaluations: List[TrialEvaluations],
) -> List[Tuple[int, int, float]]:
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


def _create_trials_evaluations(
    data: List[Tuple[int, int, float]]
) -> List[TrialEvaluations]:
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
    total_size = sum(trial_eval.num_cases() for trial_eval in state.trials_evaluations)
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
