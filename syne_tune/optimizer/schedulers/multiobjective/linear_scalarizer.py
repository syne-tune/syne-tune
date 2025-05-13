import logging
import string
from collections.abc import Iterable
from itertools import groupby
import random
from typing import Dict, Any, List, Union, Optional

import numpy as np

from syne_tune.config_space import config_space_to_json_dict
from syne_tune.util import dump_json_with_numpy
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)
MAX_NAME_LENGTH = 64
RSTRING_LENGTH = 10


def _all_equal(iterable: Iterable) -> bool:
    """
    Check if all elements of an iterable are the same
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class LinearScalarizedScheduler(TrialScheduler):
    """Scheduler with linear scalarization of multiple objectives

    This method optimizes a single objective equal to the linear scalarization
    of given two objectives. The scalarized single objective is named:
    ``"scalarized_<metric1>_<metric2>_..._<metricN>"``.
    :param config_space: Configuration space for evaluation function
    :param metric: Names of metrics to optimize
    :param do_minimize: True if we minimize the objective function
    :param scalarization_weights: Weights used to scalarize objectives. Defaults to
        an array of 1s
    :param base_scheduler_kwargs: Additional arguments to ``base_scheduler_factory``
        beyond ``config_space``, ``metric``, ``mode``
    """

    scalarization_weights: np.ndarray
    single_objective_metric: str
    base_scheduler: SingleObjectiveScheduler

    def __init__(
        self,
        config_space: Dict[str, Any],
        metrics: List[str],
        do_minimize: Optional[bool] = True,
        scalarization_weights: Union[np.ndarray, List[float]] = None,
        random_seed: int = None,
        **base_scheduler_kwargs,
    ):
        super(LinearScalarizedScheduler, self).__init__(random_seed=random_seed)
        if scalarization_weights is None:
            scalarization_weights = np.ones(shape=len(metrics))
        self.scalarization_weights = np.asarray(scalarization_weights)

        self.metrics = metrics
        self.config_space = config_space
        self.do_minimize = do_minimize

        assert (
            len(metrics) > 1
        ), "This Scheduler is intended for multi-objective optimization but only one metric is provided"
        self.single_objective_metric = f"scalarized_{'_'.join(metrics)}"
        if len(self.single_objective_metric) > MAX_NAME_LENGTH:
            # In case of multiple objectives, the name can become too long and clutter logs/results
            # If that is the case, we replace the objective names with a random string
            # to make it short but avoid collision with other results
            rstring = "".join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits)
                for _ in range(RSTRING_LENGTH)
            )
            self.single_objective_metric = f"scalarized_objective_{rstring}"

        self.base_scheduler = SingleObjectiveScheduler(
            config_space=config_space,
            metric=self.single_objective_metric,
            do_minimize=do_minimize,
            random_seed=random_seed,
            **base_scheduler_kwargs,
        )

    def _scalarized_metric(self, result: Dict[str, Any]) -> float:
        mo_results = np.array([result[item] for item in self.metrics])
        return np.sum(np.multiply(mo_results, self.scalarization_weights))

    def suggest(self) -> Optional[TrialSuggestion]:
        """Implements ``suggest``, except for basic postprocessing of config.
        See the docstring of the chosen base_scheduler for details
        """
        return self.base_scheduler.suggest()

    def on_trial_add(self, trial: Trial):
        """Called when a new trial is added to the trial runner.
        See the docstring of the chosen base_scheduler for details
        """
        return self.base_scheduler.on_trial_add(trial)

    def on_trial_error(self, trial: Trial):
        """Called when a trial has failed.
        See the docstring of the chosen base_scheduler for details
        """
        return self.base_scheduler.on_trial_error(trial)

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """Called on each intermediate result reported by a trial.
        See the docstring of the chosen base_scheduler for details
        """
        local_results = {
            self.single_objective_metric: self._scalarized_metric(result),
            **result,
        }
        return self.base_scheduler.on_trial_result(trial, local_results)

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Notification for the completion of trial.
        See the docstring of the chosen base_scheduler for details
        """
        local_results = {
            self.single_objective_metric: self._scalarized_metric(result),
            **result,
        }
        return self.base_scheduler.on_trial_complete(trial, local_results)

    def on_trial_remove(self, trial: Trial):
        """Called to remove trial.
        See the docstring of the chosen base_scheduler for details
        """
        return self.base_scheduler.on_trial_remove(trial)

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata of the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        metadata["scalarized_metric"] = self.single_objective_metric
        return metadata

    def metric_names(self) -> List[str]:
        return self.metrics
