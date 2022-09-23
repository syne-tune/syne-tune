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
from collections import Counter
from typing import Callable
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    PendingEvaluation,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)


@dataclass
class MapReward:
    forward: Callable[[float], float]
    reverse: Callable[[float], float]

    def __call__(self, x: float) -> float:
        return self.forward(x)


def map_reward_const_minus_x(const=1.0) -> MapReward:
    """
    Factory for map_reward argument in GPMultiFidelitySearcher.
    """

    def const_minus_x(x):
        return const - x

    return MapReward(forward=const_minus_x, reverse=const_minus_x)


SUPPORTED_INITIAL_SCORING = {"thompson_indep", "acq_func"}


DEFAULT_INITIAL_SCORING = "thompson_indep"


def encode_state(state: TuningJobState) -> dict:
    trials_evaluations = [
        {"trial_id": x.trial_id, "metrics": x.metrics} for x in state.trials_evaluations
    ]
    pending_evaluations = [
        {"trial_id": x.trial_id, "resource": x.resource}
        if x.resource is not None
        else {"trial_id": x.trial_id}
        for x in state.pending_evaluations
    ]
    enc_state = {
        "config_for_trial": state.config_for_trial,
        "trials_evaluations": trials_evaluations,
        "failed_trials": state.failed_trials,
        "pending_evaluations": pending_evaluations,
    }
    return enc_state


def decode_state(enc_state: dict, hp_ranges: HyperparameterRanges) -> TuningJobState:
    trials_evaluations = [
        TrialEvaluations(**x) for x in enc_state["trials_evaluations"]
    ]
    pending_evaluations = [
        PendingEvaluation(**x) for x in enc_state["pending_evaluations"]
    ]
    return TuningJobState(
        hp_ranges=hp_ranges,
        config_for_trial=enc_state["config_for_trial"],
        trials_evaluations=trials_evaluations,
        failed_trials=enc_state["failed_trials"],
        pending_evaluations=pending_evaluations,
    )


def _get_trial_id(
    hp_ranges: HyperparameterRanges,
    config: dict,
    config_for_trial: dict,
    trial_for_config: dict,
) -> str:
    match_str = hp_ranges.config_to_match_string(config, skip_last=True)
    trial_id = trial_for_config.get(match_str)
    if trial_id is None:
        trial_id = str(len(trial_for_config))
        trial_for_config[match_str] = trial_id
        config_for_trial[trial_id] = config
    return trial_id


def decode_state_from_old_encoding(
    enc_state: dict, hp_ranges: HyperparameterRanges
) -> TuningJobState:
    """
    Decodes `TuningJobState` from encoding done for the old definition of
    `TuningJobState`. Code maintained for backwards compatibility.

    Note: Since the old `TuningJobState` did not contain `trial_id`, we need
    to make them up here. We assign these IDs in the order
    `candidate_evaluations`, `failed_candidates`, `pending_candidates`,
    matching for duplicates.

    :param enc_state:
    :param hp_ranges:
    :return:
    """
    config_for_trial = dict()
    trial_for_config = dict()
    trials_evaluations = []
    for cand_eval in enc_state["candidate_evaluations"]:
        config = cand_eval["candidate"]
        trial_id = _get_trial_id(hp_ranges, config, config_for_trial, trial_for_config)
        trials_evaluations.append(TrialEvaluations(trial_id, cand_eval["metrics"]))
    failed_trials = []
    for failed_cand in enc_state["failed_candidates"]:
        failed_trials.append(
            _get_trial_id(hp_ranges, failed_cand, config_for_trial, trial_for_config)
        )
    pending_evaluations = []
    resource_attr_name = hp_ranges.name_last_pos
    for pending_cand in enc_state["pending_candidates"]:
        resource = None
        if resource_attr_name is not None and resource_attr_name in pending_cand:
            # Extended config (multi-fidelity)
            resource = int(pending_cand[resource_attr_name])
            pending_cand = pending_cand.copy()
            del pending_cand[resource_attr_name]
        trial_id = _get_trial_id(
            hp_ranges, pending_cand, config_for_trial, trial_for_config
        )
        pending_evaluations.append(PendingEvaluation(trial_id, resource))
    return TuningJobState(
        hp_ranges=hp_ranges,
        config_for_trial=config_for_trial,
        trials_evaluations=trials_evaluations,
        failed_trials=failed_trials,
        pending_evaluations=pending_evaluations,
    )


class ResourceForAcquisitionMap:
    """
    In order to use a standard acquisition function (like expected improvement)
    for multi-fidelity HPO, we need to decide at which `r_acq` we would like
    to evaluate the AF, w.r.t. the posterior distribution over `f(x, r=r_acq)`.
    This decision can depend on the current state.

    """

    def __call__(self, state: TuningJobState, **kwargs) -> int:
        raise NotImplementedError()


class ResourceForAcquisitionBOHB(ResourceForAcquisitionMap):
    """
    Implements a heuristic proposed in the BOHB paper: `r_acq` is the
    largest `r` such that we have at least `threshold` observations at
    `r`. If there are less than `threshold` observations at all levels,
    the smallest level is returned.

    """

    def __init__(self, threshold: int, active_metric: str = INTERNAL_METRIC_NAME):
        self.threshold = threshold
        self.active_metric = active_metric

    def __call__(self, state: TuningJobState, **kwargs) -> int:
        assert (
            state.num_observed_cases(self.active_metric) > 0
        ), f"state must have data for metric {self.active_metric}"
        all_resources = []
        for cand_eval in state.trials_evaluations:
            all_resources += [
                int(r) for r in cand_eval.metrics[self.active_metric].keys()
            ]
        histogram = Counter(all_resources)
        return self._max_at_least_threshold(histogram)

    def _max_at_least_threshold(self, counter: Counter) -> int:
        """
        Get largest key of `counter` whose value is at least `threshold`.

        :param counter: dict with keys that support comparison operators
        :return: largest key of `counter`
        """
        return max(
            filter(lambda r: counter[r] >= self.threshold, counter.keys()),
            default=min(counter.keys()),
        )


class ResourceForAcquisitionFirstMilestone(ResourceForAcquisitionMap):
    """
    Here, `r_acq` is the smallest rung level to be attained by a config
    started from scratch.

    """

    def __call__(self, state: TuningJobState, **kwargs) -> int:
        assert "milestone" in kwargs, (
            "Need the first milestone to be attained by the new config "
            + "passed as kwargs['milestone']. Use a scheduler which does "
            + "that (e.g., HyperbandScheduler)"
        )
        return kwargs["milestone"]


class ResourceForAcquisitionFinal(ResourceForAcquisitionMap):
    """
    Here, `r_acq = r_max` is the largest resource level.

    """

    def __init__(self, r_max: int):
        self._r_max = r_max

    def __call__(self, state: TuningJobState, **kwargs) -> int:
        return self._r_max


SUPPORTED_RESOURCE_FOR_ACQUISITION = {"bohb", "first", "final"}


def resource_for_acquisition_factory(
    kwargs: dict, hp_ranges: HyperparameterRanges
) -> ResourceForAcquisitionMap:
    resource_acq = kwargs.get("resource_acq", "bohb")
    assert (
        resource_acq in SUPPORTED_RESOURCE_FOR_ACQUISITION
    ), f"resource_acq = {resource_acq} not supported, must be in " + str(
        SUPPORTED_RESOURCE_FOR_ACQUISITION
    )
    if resource_acq == "bohb":
        threshold = kwargs.get("resource_acq_bohb_threshold", hp_ranges.ndarray_size)
        resource_for_acquisition = ResourceForAcquisitionBOHB(threshold=threshold)
    elif resource_acq == "first":
        assert resource_acq == "first", "resource_acq must be 'bohb' or 'first'"
        resource_for_acquisition = ResourceForAcquisitionFirstMilestone()
    else:
        r_max = kwargs["max_epochs"]
        resource_for_acquisition = ResourceForAcquisitionFinal(r_max=r_max)
    return resource_for_acquisition
