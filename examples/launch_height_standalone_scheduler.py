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
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, \
    SchedulerDecision, TrialSuggestion
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint


class SimpleScheduler(TrialScheduler):
    def __init__(self, config_space: Dict, metric: str):
        super(SimpleScheduler, self).__init__(config_space=config_space)
        self.metric = metric
        self.sorted_results = []

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        # Called when a slot exists to run a trial, here we simply draw a
        # random candidate.
        config = {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }
        return TrialSuggestion.start_suggestion(config)

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        # Given a new result, we decide whether the trial should stop or continue.
        # In this case, we implement a naive strategy that stops if the result is worse than 80% of previous results.
        # This is a naive strategy as we do not account for the fact that trial improves with more steps.

        new_metric = result[self.metric]

        # insert new metric in sorted results
        index = np.searchsorted(self.sorted_results, new_metric)
        self.sorted_results = np.insert(self.sorted_results, index, new_metric)
        normalized_rank = index / float(len(self.sorted_results))

        if normalized_rank < 0.8:
            return SchedulerDecision.CONTINUE
        else:
            logging.info(
                f"see new results {new_metric} which rank {normalized_rank * 100}%, "
                f"stopping it as it does not rank on the top 80%"
            )
            return SchedulerDecision.STOP

    def metric_names(self) -> List[str]:
        return [self.metric]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = str(
        Path(__file__).parent / "training_scripts" / "height_example" /
        "train_height.py")
    metric = "mean_loss"

    # Local back-end
    trial_backend = LocalBackend(entry_point=entry_point)

    np.random.seed(random_seed)
    scheduler = SimpleScheduler(
        config_space=config_space,
        metric=metric)

    stop_criterion = StoppingCriterion(max_wallclock_time=30)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
