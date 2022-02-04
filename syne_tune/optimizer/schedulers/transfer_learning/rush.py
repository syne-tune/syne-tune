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
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations, TransferLearningScheduler


class RUSHScheduler(TransferLearningScheduler):
    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric: str,
            **kwargs
    ) -> None:
        super().__init__(config_space=config_space,
                         transfer_learning_evaluations=transfer_learning_evaluations,
                         metric_names=[metric])
        points_to_evaluate = RUSHScheduler._determine_baseline_configurations(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric=metric,
            mode=kwargs.get('mode', 'min')
        )
        if 'points_to_evaluate' in kwargs:
            points_to_evaluate += kwargs['points_to_evaluate']
            points_to_evaluate = [dict(s) for s in set(frozenset(p.items()) for p in points_to_evaluate)]
        kwargs['points_to_evaluate'] = points_to_evaluate

        self._hyperband_scheduler = HyperbandScheduler(config_space, **kwargs)
        self._num_init_configs = len(points_to_evaluate)
        self._thresholds = dict()  # thresholds at different resource levels that must be met

    @staticmethod
    def _determine_baseline_configurations(config_space: Dict,
                                           transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
                                           metric: str,
                                           mode: str) -> List[Dict]:
        argbest, best = (np.argmin, np.min) if mode == 'min' else (np.argmax, np.max)
        baseline_configurations = list()
        for evals in transfer_learning_evaluations.values():
            best_hpc_idx = argbest(best(evals.objective_values(objective_name=metric), axis=1))
            hpc = evals.hyperparameters.iloc[best_hpc_idx]
            baseline_configurations.append({
                key: hpc[key] for key in config_space
            })
        return baseline_configurations

    def on_trial_error(self, trial: Trial) -> None:
        self._hyperband_scheduler.on_trial_error(trial)

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        trial_decision = self._hyperband_scheduler.on_trial_result(trial, result)
        return self._on_trial_result(trial_decision, trial, result)

    def on_trial_remove(self, trial: Trial) -> None:
        self._hyperband_scheduler.on_trial_remove(trial)

    def on_trial_complete(self, trial: Trial, result: Dict) -> None:
        self._hyperband_scheduler.on_trial_complete(trial, result)

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        return self._hyperband_scheduler.suggest(trial_id)

    def state_dict(self) -> Dict:
        record = self._hyperband_scheduler.state_dict()
        record['num_init_configs'] = pickle.dumps(self._num_init_configs)
        record['thresholds'] = pickle.dumps(self._thresholds)
        return record

    def load_state_dict(self, state_dict):
        self._num_init_configs = pickle.loads(state_dict['num_init_configs'])
        del state_dict['num_init_configs']
        self._thresholds = pickle.loads(state_dict['thresholds'])
        del state_dict['thresholds']
        self._hyperband_scheduler(state_dict)

    def _on_trial_result(self, trial_decision: str, trial: Trial, result: Dict) -> str:
        if trial_decision != SchedulerDecision.CONTINUE:
            return trial_decision

        metric_val = float(result[self._hyperband_scheduler.metric])
        resource = int(result[self._hyperband_scheduler._resource_attr])
        trial_id = str(trial.trial_id)

        if self._is_milestone_reached(trial_id, resource):
            if self._is_in_points_to_evaluate(trial):
                self._thresholds[resource] = self._return_better(self._thresholds.get(resource),
                                                                 metric_val)
            elif not self._meets_threshold(metric_val, resource):
                return SchedulerDecision.STOP

        return trial_decision

    def _is_milestone_reached(self, trial_id: str, resource: int) -> bool:
        rung_sys, bracket_id, skip_rungs = self._hyperband_scheduler.terminator._get_rung_system(trial_id)
        rungs = rung_sys._rungs[:(-skip_rungs if skip_rungs > 0 else None)]
        return resource in [rung.level for rung in rungs]

    def _is_in_points_to_evaluate(self, trial: Trial) -> bool:
        return int(trial.trial_id) < self._num_init_configs

    def _meets_threshold(self, metric_val: float, resource: int) -> bool:
        threshold = self._thresholds.get(resource)
        if threshold is None:
            return True
        if self.metric_mode() == 'min':
            return metric_val <= threshold
        else:
            return metric_val >= threshold

    def _return_better(self, val1: float, val2: float) -> bool:
        if self.metric_mode() == 'min':
            better_val = min(float('inf') if val1 is None else val1,
                             float('inf') if val2 is None else val2)
        else:
            better_val = max(float('-inf') if val1 is None else val1,
                             float('-inf') if val2 is None else val2)
        return better_val
