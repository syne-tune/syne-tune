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
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    StateForModelConverter,
)


ObservedData = List[Tuple[int, float]]


def _extract_observations(
    trials_evaluations: List[TrialEvaluations],
) -> ObservedData:
    """
    Maps ``trials_evaluations`` to list of tuples :math:`(i, y_i)`, where
    :math:`y_i` is the observed value for trial ID :math:`i`.

    :param trials_evaluations: See above
    :return: See above
    """
    return [
        (int(trial_eval.trial_id), trial_eval.metrics[INTERNAL_METRIC_NAME])
        for trial_eval in trials_evaluations
    ]


def _create_trials_evaluations(data: ObservedData) -> List[TrialEvaluations]:
    """Inverse of :func:`_extract_observations`

    :param data: List of tuples
    :return: Resulting ``trials_evaluations``
    """
    return [
        TrialEvaluations(
            trial_id=str(trial_id), metrics={INTERNAL_METRIC_NAME: metric_val}
        )
        for trial_id, metric_val in data
    ]


def cap_size_tuning_job_state(
    state: TuningJobState,
    max_size: int,
    mode: str,
    top_fraction: float,
    random_state: Optional[RandomState] = None,
) -> TuningJobState:
    """
    Returns state which is identical to ``state``, except that the
    ``trials_evaluations`` are replaced by a subset so the total number of
    metric values is ``<= max_size``.

    :param state: Original state to filter down
    :param max_size: Maximum number of observed metric values in new state
    :param mode: "min" or "max"
    :param top_fraction: See above
    :param random_state: Used for random sampling. Defaults to ``numpy.random``.
    :return: New state meeting the ``max_size`` constraint. This is a copy of
        ``state`` even if this meets the constraint already.
    """
    total_size = state.num_observed_cases()
    if total_size <= max_size:
        trials_evaluations = copy.deepcopy(state.trials_evaluations)
    else:
        data = sorted(
            _extract_observations(state.trials_evaluations),
            key=itemgetter(1),
            reverse=mode == "max",
        )
        n_top = int(round(max_size * top_fraction))
        new_data = data[:n_top]
        n_rem = max_size - n_top
        if n_rem > 0:
            index = random_state.choice(total_size - n_top, size=n_rem, replace=False)
            new_data += [data[n_top + i] for i in index]
        trials_evaluations = _create_trials_evaluations(new_data)
    return TuningJobState(
        hp_ranges=state.hp_ranges,
        config_for_trial=state.config_for_trial.copy(),
        trials_evaluations=trials_evaluations,
        failed_trials=state.failed_trials.copy(),
        pending_evaluations=state.pending_evaluations.copy(),
    )


class SubsampleSingleFidelityStateConverter(StateForModelConverter):
    """
    Converts state by (possibly) down sampling the observation so that their
    total number is ``<= max_size``. If ``len(state) > max_size``, the subset
    is sampled as follows. ``max_size * top_fraction`` is filled with the best
    observations. The remainder is sampled without replacement from the
    remaining observations.

    :param max_size: Maximum number of observed metric values in new state
    :param mode: "min" or "max"
    :param top_fraction: See above
    :param random_state: Used for random sampling. Can also be set with
        :meth:`set_random_state`
    """

    def __init__(
        self,
        max_size: int,
        mode: str,
        top_fraction: float,
        random_state: Optional[RandomState] = None,
    ):
        support_mode = ["min", "max"]
        assert (
            mode in support_mode
        ), f"mode = {mode} not supported, must be in {support_mode}"
        assert (
            0 <= top_fraction <= 1
        ), f"top_fraction = {top_fraction} must be in [0, 1]"
        self.max_size = int(max_size)
        assert self.max_size >= 1
        self._random_state = random_state
        self._mode = mode
        self._top_fraction = top_fraction

    def __call__(self, state: TuningJobState) -> TuningJobState:
        assert (
            self._random_state is not None
        ), "Call set_random_state before first usage"
        return cap_size_tuning_job_state(
            state=state,
            max_size=self.max_size,
            mode=self._mode,
            top_fraction=self._top_fraction,
            random_state=self._random_state,
        )

    def set_random_state(self, random_state: RandomState):
        self._random_state = random_state
