import logging
from typing import Optional, List, Dict, Any

import numpy as np
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
    NonDominatedPriority,
)
from syne_tune.config_space import config_space_to_json_dict
from syne_tune.util import dump_json_with_numpy
from syne_tune.optimizer.schedulers.asha import Bracket

logger = logging.getLogger(__name__)


class MOASHA(TrialScheduler):
    """
    Implements MultiObjective Asynchronous Successive HAlving with different
    multiobjective sort options. References:

        | A multi-objective perspective on jointly tuning hardware and hyperparameters
        | David Salinas, Valerio Perrone, Cedric Archambeau and Olivier Cruchant
        | NAS workshop, ICLR2021.

    and

        | Multi-objective multi-fidelity hyperparameter optimization with application to fairness
        | Robin Schmucker, Michele Donini, Valerio Perrone, Cédric Archambeau

    :param config_space: Configuration space
    :param metrics: List of metric names MOASHA optimizes over
    :param do_minimize: If True, we minimize the objective function specified by ``metric`` . Defaults to True.
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param multiobjective_priority: The multiobjective priority that is used
        to sort multiobjective candidates. We support several choices such
        as non-dominated sort or linear scalarization, default is
        non-dominated sort.
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param grace_period: Only stop trials at least this old in time.
        The units are the same as the attribute named by ``time_attr``.
        Defaults to 1
    :param reduction_factor: Used to set halving rate and amount. This
        is simply a unit-less scalar. Defaults to 3
    :param brackets: Number of brackets. Each bracket has a different
        ``grace_period`` and number of rung levels. Defaults to 1
    :param random_seed: Seed for the random number generator
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metrics: List[str],
        do_minimize: Optional[bool] = True,
        time_attr: str = "training_iteration",
        multiobjective_priority: Optional[MOPriority] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 3,
        brackets: int = 1,
        random_seed: int = None,
    ):
        super(MOASHA, self).__init__(random_seed=random_seed)
        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "reduction factor not valid!"
        assert brackets > 0, "brackets must be positive!"

        if multiobjective_priority is None:
            self.multiobjective_priority = NonDominatedPriority()
        else:
            self.multiobjective_priority = multiobjective_priority

        self.config_space = config_space
        self.do_minimize = do_minimize
        self.reduction_factor = reduction_factor
        self.max_t = max_t
        self.trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self.brackets = [
            Bracket(
                grace_period, max_t, reduction_factor, s, self.multiobjective_priority
            )
            for s in range(brackets)
        ]
        self.num_stopped = 0
        self.metrics = metrics
        self.metric_multiplier = 1 if self.do_minimize else -1
        self.time_attr = time_attr

    def metric_names(self) -> List[str]:
        return self.metrics

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"

    def suggest(self) -> Optional[TrialSuggestion]:
        config = {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }
        return TrialSuggestion.start_suggestion(config)

    def on_trial_add(self, trial: Trial):
        sizes = np.array([len(b.rungs) for b in self.brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self.brackets), p=normalized)
        self.trial_info[trial.trial_id] = self.brackets[idx]

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        self.check_metrics_are_present(result)
        if result[self.time_attr] >= self.max_t:
            action = SchedulerDecision.STOP
        else:
            bracket = self.trial_info[trial.trial_id]
            metrics = self.metric_dict(result)
            action = bracket.on_result(
                trial_id=trial.trial_id,
                cur_iter=result[self.time_attr],
                metrics=metrics,
            )
        if action == SchedulerDecision.STOP:
            self.num_stopped += 1
        return action

    def metric_dict(self, reported_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            metric: reported_results[metric] * self.metric_multiplier
            for metric in self.metrics
        }

    def check_metrics_are_present(self, result: Dict[str, Any]):
        for key in [self.time_attr] + self.metrics:
            if key not in result:
                assert key in result, f"{key} not found in reported result {result}"

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        self.check_metrics_are_present(result)
        bracket = self.trial_info[trial.trial_id]
        bracket.on_result(
            trial_id=trial.trial_id,
            cur_iter=result[self.time_attr],
            metrics=self.metric_dict(result),
        )
        del self.trial_info[trial.trial_id]

    def on_trial_remove(self, trial: Trial):
        del self.trial_info[trial.trial_id]

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        metadata["metrics"] = self.metrics
        metadata["metric_names"] = self.metric_names()
        metadata["metric_mode"] = self.metric_mode()
        return metadata
