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
from typing import List, Dict, Optional

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    PendingEvaluation,
    MetricValues,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)


class TuningJobState:
    """
    Collects all data determining the state of a tuning experiment. Trials
    are indexed by `trial_id`. The configurations associated with trials are
    listed in `config_for_trial`.
    `trials_evaluations` contains observations, `failed_trials` lists
    trials for which evaluations have failed, `pending_evaluations` lists
    trials for which observations are pending.

    `trials_evaluations` may store values for different metrics in each
    record, and each such value may be a dict (see:class:`TrialEvaluations`).
    For example, for multi-fidelity schedulers,
    `trials_evaluations[i].metrics[k][str(r)]` is the value for metric k
    and trial `trials_evaluations[i].trial_id` observed at resource level
    r.
    """

    def __init__(
        self,
        hp_ranges: HyperparameterRanges,
        config_for_trial: Dict[str, Configuration],
        trials_evaluations: List[TrialEvaluations],
        failed_trials: List[str] = None,
        pending_evaluations: List[PendingEvaluation] = None,
    ):
        if failed_trials is None:
            failed_trials = []
        if pending_evaluations is None:
            pending_evaluations = []
        self._check_trial_ids(
            config_for_trial, trials_evaluations, failed_trials, pending_evaluations
        )
        self.hp_ranges = hp_ranges
        self.config_for_trial = config_for_trial
        self.trials_evaluations = trials_evaluations
        self.failed_trials = failed_trials
        self.pending_evaluations = pending_evaluations

    @staticmethod
    def _check_all_string(trial_ids: List[str], name: str):
        assert all(
            isinstance(x, str) for x in trial_ids
        ), f"trial_ids in {name} contain non-string values:\n{trial_ids}"

    @staticmethod
    def _check_trial_ids(
        config_for_trial, trials_evaluations, failed_trials, pending_evaluations
    ):
        observed_trials = [x.trial_id for x in trials_evaluations]
        pending_trials = [x.trial_id for x in pending_evaluations]
        TuningJobState._check_all_string(observed_trials, "trials_evaluations")
        TuningJobState._check_all_string(failed_trials, "failed_trials")
        TuningJobState._check_all_string(pending_trials, "pending_evaluations")
        trial_ids = set(observed_trials + failed_trials + pending_trials)
        for trial_id in trial_ids:
            assert (
                trial_id in config_for_trial
            ), f"trial_id {trial_id} not contained in configs_for_trials"

    @staticmethod
    def empty_state(hp_ranges: HyperparameterRanges) -> "TuningJobState":
        return TuningJobState(
            hp_ranges=hp_ranges,
            config_for_trial=dict(),
            trials_evaluations=[],
            failed_trials=[],
            pending_evaluations=[],
        )

    def _find_labeled(self, trial_id: str) -> int:
        try:
            return next(
                i
                for i, x in enumerate(self.trials_evaluations)
                if x.trial_id == trial_id
            )
        except StopIteration:
            return -1

    def _find_pending(self, trial_id: str, resource: Optional[int] = None) -> int:
        try:
            return next(
                i
                for i, x in enumerate(self.pending_evaluations)
                if x.trial_id == trial_id and x.resource == resource
            )
        except StopIteration:
            return -1

    def _register_config_for_trial(
        self, trial_id: str, config: Optional[Configuration] = None
    ):
        if config is None:
            assert trial_id in self.config_for_trial, (
                f"trial_id = {trial_id} not yet registered in "
                + "config_for_trial, so config must be given"
            )
        elif trial_id not in self.config_for_trial:
            self.config_for_trial[trial_id] = config.copy()

    def metrics_for_trial(
        self, trial_id: str, config: Optional[Configuration] = None
    ) -> MetricValues:
        """
        Helper for inserting new entry into `trials_evaluations`. If `trial_id`
        is already contained there, the corresponding `eval.metrics` is
        returned. Otherwise, a new entry `new_eval` is appended to
        `trials_evaluations` and its `new_eval.metrics` is returned
        (empty dict). In the latter case, `config` needs to be passed,
        because it may not yet feature in `config_for_trial`.

        """
        # NOTE: If `trial_id` exists in `config_for_trial` and `config` is
        # given, we do not check that `config` is correct. In fact, we ignore
        # `config` in this case.
        self._register_config_for_trial(trial_id, config)
        pos = self._find_labeled(trial_id)
        if pos != -1:
            metrics = self.trials_evaluations[pos].metrics
        else:
            # New entry
            metrics = dict()
            new_eval = TrialEvaluations(trial_id=trial_id, metrics=metrics)
            self.trials_evaluations.append(new_eval)
        return metrics

    def num_observed_cases(self, metric_name: str = INTERNAL_METRIC_NAME) -> int:
        return sum(ev.num_cases(metric_name) for ev in self.trials_evaluations)

    def observed_data_for_metric(
        self, metric_name: str = INTERNAL_METRIC_NAME, resource_attr_name: str = None
    ) -> (List[Configuration], List[float]):
        """
        Extracts datapoints from `trials_evaluations` for particular
        metric `metric_name`, in the form of a list of configs and a list of
        metric values.
        If `metric_name` is a dict-valued metric, the dict keys must be
        resource values, and the returned configs are extended. Here, the
        name of the resource attribute can be passed in `resource_attr_name`
        (if not given, it can be obtained from `hp_ranges` if this is extended).

        Note: Implements the default behaviour, namely to return extended
        configs for dict-valued metrics, which also require `hp_ranges` to be
        extended. This is not correct for some specific multi-fidelity
        surrogate models, which should access the data directly.

        :param metric_name:
        :param resource_attr_name:
        :return: configs, metric_values
        """
        if resource_attr_name is None:
            resource_attr_name = self.hp_ranges.name_last_pos
        configs = []
        metric_values = []
        for ev in self.trials_evaluations:
            config = self.config_for_trial[ev.trial_id]
            metric_entry = ev.metrics.get(metric_name)
            if metric_entry is not None:
                if isinstance(metric_entry, dict):
                    assert resource_attr_name is not None, (
                        "Need resource_attr_name for dict-valued metric " + metric_name
                    )
                    for resource, metric_val in metric_entry.items():
                        config_ext = dict(config, **{resource_attr_name: int(resource)})
                        configs.append(config_ext)
                        metric_values.append(metric_val)
                else:
                    configs.append(config)
                    metric_values.append(metric_entry)
        return configs, metric_values

    def is_pending(self, trial_id: str, resource: Optional[int] = None) -> bool:
        return self._find_pending(trial_id, resource) != -1

    def is_labeled(
        self,
        trial_id: str,
        metric_name: str = INTERNAL_METRIC_NAME,
        resource: Optional[int] = None,
    ) -> bool:
        """
        Checks whether `trial_id` has observed data under `metric_name`. If
        `resource` is given, the observation must be at that resource level.

        """
        pos = self._find_labeled(trial_id)
        result = False
        if pos != -1:
            metric_entry = self.trials_evaluations[pos].metrics.get(metric_name)
            if metric_entry is not None:
                if resource is None:
                    result = True
                elif isinstance(metric_entry, dict):
                    result = str(resource) in metric_entry
        return result

    def append_pending(
        self,
        trial_id: str,
        config: Optional[Configuration] = None,
        resource: Optional[int] = None,
    ):
        """
        Appends new pending evaluation. If the trial has not been registered
        here, `config` must be given. Otherwise, it is ignored.

        """
        self._register_config_for_trial(trial_id, config)
        assert not self.is_pending(trial_id, resource)
        self.pending_evaluations.append(
            PendingEvaluation(trial_id=trial_id, resource=resource)
        )

    def remove_pending(self, trial_id: str, resource: Optional[int] = None) -> bool:
        pos = self._find_pending(trial_id, resource)
        if pos != -1:
            self.pending_evaluations.pop(pos)
            return True
        else:
            return False

    def pending_configurations(
        self, resource_attr_name: str = None
    ) -> List[Configuration]:
        """
        Returns list of configurations corresponding to pending evaluations.
        If the latter have resource values, the configs are extended.

        """
        if resource_attr_name is None:
            resource_attr_name = self.hp_ranges.name_last_pos
        configs = []
        for pend_eval in self.pending_evaluations:
            config = self.config_for_trial[pend_eval.trial_id]
            resource = pend_eval.resource
            if resource is not None:
                assert (
                    resource_attr_name is not None
                ), f"Need resource_attr_name, or hp_ranges to be extended"
                config = dict(config, **{resource_attr_name: int(resource)})
            configs.append(config)
        return configs

    def _map_configs_for_matching(
        self, config_for_trial: Dict[str, Configuration]
    ) -> Dict[str, str]:
        return {
            trial_id: self.hp_ranges.config_to_match_string(config)
            for trial_id, config in config_for_trial.items()
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, TuningJobState):
            return False
        if (
            self.failed_trials != other.failed_trials
            or self.pending_evaluations != other.pending_evaluations
        ):
            return False
        if self.hp_ranges != other.hp_ranges:
            return False
        if self.trials_evaluations != other.trials_evaluations:
            return False
        return self._map_configs_for_matching(
            self.config_for_trial
        ) == self._map_configs_for_matching(other.config_for_trial)
