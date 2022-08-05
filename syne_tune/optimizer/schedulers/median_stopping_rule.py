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
"""
Example showing how to implement a new Scheduler.
"""
import logging
from collections import defaultdict
from typing import Optional, Dict, List

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    SchedulerDecision,
    TrialSuggestion,
)


class MedianStoppingRule(TrialScheduler):
    def __init__(
        self,
        scheduler: TrialScheduler,
        resource_attr: str,
        running_average: bool = True,
        metric: Optional[str] = None,
        grace_time: Optional[int] = 1,
        grace_population: int = 5,
        rank_cutoff: float = 0.5,
    ):
        """
        Applies median stopping rule in top of an existing scheduler.
        * If result at time-step ranks less than the cutoff of other results observed at this time-step, the trial is
        interrupted and otherwise, the wrapped scheduler is called to make the stopping decision.
        * Suggest decisions are left to the wrapped scheduler.
        * The mode of the wrapped scheduler is used.
        Reference: Google Vizier: A Service for Black-Box Optimization. Golovin et al. 2017.
        :param scheduler: scheduler to be called for trial suggestion or when median-stopping-rule decision is to
        continue.
        :param resource_attr: key in the reported dictionary that accounts for the resource (e.g. epoch or
        wall-clocktime).
        :param running_average: if True, then uses the running average of observation instead of raw observations.
        :param metric: metric to be considered.
        :param grace_time: median stopping rule is only applied for results whose `time_attr` exceeds this amount.
        :param grace_population: median stopping rule when at least `grace_population` have been observed at a resource
        level.
        :param rank_cutoff: results whose quantiles are bellow this level are discarded (discard by default trials
         whose results are bellow the median).
        """
        super(MedianStoppingRule, self).__init__(config_space=scheduler.config_space)
        self.metric = scheduler.metric if metric is None else metric
        self.sorted_results = defaultdict(list)
        self.scheduler = scheduler
        self.resource_attr = resource_attr
        self.rank_cutoff = rank_cutoff
        self.grace_time = grace_time
        self.min_samples_required = grace_population
        self.running_average = running_average
        if running_average:
            self.trial_to_results = defaultdict(list)
        self.mode = scheduler.metric_mode()

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        return self.scheduler._suggest(trial_id=trial_id)

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        new_metric = result[self.metric]
        if self.mode == "max":
            new_metric *= -1
        time_step = result[self.resource_attr]

        if self.running_average:
            # gets the running average of current observations
            self.trial_to_results[trial.trial_id].append(new_metric)
            new_metric = np.mean(self.trial_to_results[trial.trial_id])

        # insert new metric in sorted results acquired at this resource
        index = np.searchsorted(self.sorted_results[time_step], new_metric)
        self.sorted_results[time_step] = np.insert(
            self.sorted_results[time_step], index, new_metric
        )
        normalized_rank = index / float(len(self.sorted_results[time_step]))

        if self.grace_condition(time_step=time_step):
            return self.scheduler.on_trial_result(trial=trial, result=result)
        elif normalized_rank <= self.rank_cutoff:
            return self.scheduler.on_trial_result(trial=trial, result=result)
        else:
            logging.info(
                f"see new results {new_metric} at time-step {time_step} for trial {trial.trial_id}"
                f" with rank {int(normalized_rank * 100)}%, "
                f"stopping it as it does not rank on the top {int(self.rank_cutoff * 100)}%"
            )
            return SchedulerDecision.STOP

    def grace_condition(self, time_step: float) -> bool:
        # lets the trial continue when the time is bellow the grace time and when not sufficiently many observations
        # are present for this time budget
        if (
            self.min_samples_required is not None
            and len(self.sorted_results[time_step]) < self.min_samples_required
        ):
            return True
        if self.grace_time is not None and time_step < self.grace_time:
            return True
        return False

    def metric_names(self) -> List[str]:
        return self.scheduler.metric_names()

    def metric_mode(self) -> str:
        return self.scheduler.metric_mode()
