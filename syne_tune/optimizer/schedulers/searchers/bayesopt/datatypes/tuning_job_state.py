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

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration, CandidateEvaluation, PendingEvaluation, \
    MetricValues, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges


class TuningJobState(object):
    """
    Collects all data determining the state of a tuning experiment.
    `candidate_evaluations` contains observations, `failed_candidates` lists
    configurations for which evaluations have failed, `pending_evaluations`
    stores configurations for currently running trials, for which
    observations are pending (plus, optionally, additional data such as
    fantasy samples).

    `candidate_evaluations` may store values for different metrics in each
    record (i.e., each candidate), and each such value may be a dict (see
    :class:`CandidateEvaluation`). For example, for multi-fidelity schedulers,
    `candidate_evaluations[i].metrics[k][str(r)]` is the value for metric k
    and config `candidate_evalutions[i].candidate` observed at resource level
    r.

    """
    def __init__(
            self, hp_ranges: HyperparameterRanges,
            candidate_evaluations: List[CandidateEvaluation],
            failed_candidates: List[Configuration] = None,
            pending_evaluations: List[PendingEvaluation] = None):
        self.hp_ranges = hp_ranges
        self.candidate_evaluations = candidate_evaluations
        if failed_candidates is None:
            failed_candidates = []
        self.failed_candidates = failed_candidates
        if pending_evaluations is None:
            pending_evaluations = []
        self.pending_evaluations = pending_evaluations
        self._config_pos = None

    @property
    def pending_candidates(self):
        return [x.candidate for x in self.pending_evaluations]

    def _config_key(self, config: Configuration) -> str:
        # If self.hp_ranges serves extended configs, the resource attribute
        # is skipped, so that `config` is always non-extended
        return str(self.hp_ranges.config_to_tuple(
            config, skip_last=True))

    @property
    def config_pos(self) -> Dict[str, int]:
        if self._config_pos is None:
            self._config_pos = {
                self._config_key(eval.candidate): pos
                for pos, eval in enumerate(self.candidate_evaluations)}
        return self._config_pos

    def pos_of_config(self, config: Configuration) -> Optional[int]:
        config_key = self._config_key(config)
        return self.config_pos.get(config_key)

    def metrics_for_config(self, config: Configuration) -> MetricValues:
        """
        Helper for inserting new entry into `candidate_evaluation`. If
        `config` is equal to `eval.candidate` for an entry `eval` in
        `candidate_evaluation`, the corresponding `eval.metrics` is
        returned. Otherwise, a new entry `new_eval` is appended to
        `candidate_evaluations` and its `new_eval.metrics` is returned
        (empty dict).

        """
        config_key = self._config_key(config)
        pos = self.config_pos.get(config_key)
        if pos is not None:
            return self.candidate_evaluations[pos].metrics
        else:
            new_metrics = dict()
            new_eval = CandidateEvaluation(
                candidate=config, metrics=new_metrics)
            self.config_pos[config_key] = len(self.candidate_evaluations)
            self.candidate_evaluations.append(new_eval)
            return new_metrics

    def num_observed_cases(
            self, metric_name: str = INTERNAL_METRIC_NAME) -> int:
        return sum(eval.num_cases(metric_name)
                   for eval in self.candidate_evaluations)

    def observed_data_for_metric(
            self, metric_name: str = INTERNAL_METRIC_NAME,
            resource_attr_name: str = None) -> (
            List[Configuration], List[float]):
        """
        Extracts datapoints from `candidate_evaluations` for particular
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
        if not self.candidate_evaluations:
            return [], []
        configs = []
        metric_values = []
        for eval in self.candidate_evaluations:
            config = eval.candidate
            metric_entry = eval.metrics.get(metric_name)
            if metric_entry is not None:
                if isinstance(metric_entry, dict):
                    assert resource_attr_name is not None, \
                        f"Need {resource_attr_name} for dict-valued metric " +\
                        metric_name
                    for resource, metric_val in metric_entry.items():
                        config_ext = dict(
                            config, **{resource_attr_name: int(resource)})
                        configs.append(config_ext)
                        metric_values.append(metric_val)
                else:
                    configs.append(config)
                    metric_values.append(metric_entry)
        return configs, metric_values
