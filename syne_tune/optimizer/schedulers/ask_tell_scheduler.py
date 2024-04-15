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
from typing import Dict
import datetime

from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.optimizer.scheduler import TrialScheduler


class AskTellScheduler:
    base_scheduler: TrialScheduler
    trial_counter: int
    completed_experiments: Dict[int, TrialResult]

    def __init__(self, base_scheduler: TrialScheduler):
        """
        Simple interface to use SyneTune schedulers in a custom loop, for example:

        .. code-block:: python

            scheduler = AskTellScheduler(base_scheduler=RandomSearch(config_space, metric=metric, mode=mode))
            for iter in range(max_iterations):
                trial_suggestion = scheduler.ask()
                test_result = target_function(**trial_suggestion.config)
                scheduler.tell(trial_suggestion, {metric: test_result})

        :param base_scheduler: Scheduler to be wrapped
        """
        self.base_scheduler = base_scheduler
        self.trial_counter = 0
        self.completed_experiments = {}

    def ask(self) -> Trial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self.base_scheduler.suggest(self.trial_counter)
        trial = Trial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        self.trial_counter += 1
        return trial

    def tell(self, trial: Trial, experiment_result: Dict[str, float]):
        """
        Feed experiment results back to the Scheduler

        :param trial: Trial that was run
        :param experiment_result: {metric: value} dictionary with experiment results
        """
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.datetime.now(),
        )
        self.base_scheduler.on_trial_complete(trial=trial, result=experiment_result)
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metric: str) -> TrialResult:
        """
        Return the best trial according to the provided metric.

        :param metric: Metric to use for comparison
        """
        if self.base_scheduler.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metric],
        )
