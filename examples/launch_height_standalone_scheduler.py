"""
Example showing how to implement a new Scheduler.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.legacy_scheduler import LegacyTrialScheduler
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.config_space import randint
from examples.training_scripts.height_example.train_height import (
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)


class SimpleScheduler(LegacyTrialScheduler):
    def __init__(
        self, config_space: Dict[str, Any], metric: str, mode: Optional[str] = None
    ):
        super(SimpleScheduler, self).__init__(config_space=config_space)
        self.metric = metric
        self.mode = mode if mode is not None else "min"
        self.sorted_results = []

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        # Called when a slot exists to run a trial, here we simply draw a
        # random candidate.
        config = {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }
        return TrialSuggestion.start_suggestion(config)

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        # Given a new result, we decide whether the trial should stop or continue.
        # In this case, we implement a naive strategy that stops if the result is worse than 80% of previous results.
        # This is a naive strategy as we do not account for the fact that trial improves with more steps.

        new_metric = result[self.metric]

        # insert new metric in sorted results
        index = np.searchsorted(self.sorted_results, new_metric)
        self.sorted_results = np.insert(self.sorted_results, index, new_metric)
        normalized_rank = index / float(len(self.sorted_results))

        if self.mode == "max":
            normalized_rank = 1 - normalized_rank

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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    # Local backend
    trial_backend = LocalBackend(entry_point=entry_point)

    np.random.seed(random_seed)
    scheduler = SimpleScheduler(
        config_space=config_space, metric=METRIC_ATTR, mode=METRIC_MODE
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=20)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
