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
import string
from collections.abc import Iterable
from itertools import groupby
import random
from typing import Dict, Any, List, Union, Callable, Optional

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

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
    """Scheduler with a Linear Scalarization of multiple objectives.
    This method optimizes a single objective equal to the linear scalarization of given two objectives.
    The scalarized single objective is named: 'scalarized_<metric1>_<metric2>_..._<metricN>'


    :param base_scheduler_factory: Factory method for the single-objective scheduler
        used on the scalarized objective. It will be initialized inside this Scheduler
        If None, FIFOScheduler is used.
    :param config_space: Configuration space for evaluation function
    :param metric: Names of metrics to optimize
    :param mode: Modes of metrics to optimize (min or max). All must be matching.
    :param scalarization_weights: Weights used to scalarize objectives, if None an array of 1s is used
    :param base_scheduler_kwargs: Additional arguments to base_scheduler beyond config_space, metric and mode
    """

    scalarization_weights: np.ndarray
    single_objective_metric: str
    base_scheduler: TrialScheduler

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        scalarization_weights: Union[np.ndarray, List[float]] = None,
        base_scheduler_factory: Callable[[Any], TrialScheduler] = None,
        **base_scheduler_kwargs,
    ):
        super(LinearScalarizedScheduler, self).__init__(config_space)
        if scalarization_weights is None:
            scalarization_weights = np.ones(shape=len(metric))
        self.scalarization_weights = np.asarray(scalarization_weights)

        self.metric = metric
        self.mode = mode

        assert (
            len(metric) > 1
        ), "This Scheduler is inteded for multi-objective optimization but only one metric is provided"
        self.single_objective_metric = f"scalarized_{'_'.join(metric)}"
        if len(self.single_objective_metric) > MAX_NAME_LENGTH:
            # In case of multiple objectives, the name can become too long and clutter logs/results
            # If that is the case, we replace the objective names with a random string
            # to make it short but avoid collision with other results
            rstring = "".join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits)
                for _ in range(RSTRING_LENGTH)
            )
            self.single_objective_metric = f"scalarized_objective_{rstring}"

        single_objective_mode = self.mode
        if isinstance(single_objective_mode, Iterable):
            assert len(mode) >= 1, "At least one mode must be provided"
            assert _all_equal(
                mode
            ), "Modes must be the same, use positive/negative scalarization_weights to change relative signs"
            single_objective_mode = next(x for x in mode)

        if base_scheduler_factory is None:
            base_scheduler_factory = FIFOScheduler

        self.base_scheduler = base_scheduler_factory(
            config_space=config_space,
            metric=self.single_objective_metric,
            mode=single_objective_mode,
            **base_scheduler_kwargs,
        )

    def _scalarized_metric(self, result: Dict[str, Any]) -> float:
        if isinstance(self.base_scheduler, FIFOScheduler):
            FIFOScheduler._check_keys_of_result(result, self.metric_names())

        mo_results = np.array([result[item] for item in self.metric_names()])
        return np.sum(np.multiply(mo_results, self.scalarization_weights))

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        """Implements ``suggest``, except for basic postprocessing of config.
        See the docstring of the chosen base_scheduler for details
        """
        return self.base_scheduler._suggest(trial_id)

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

    def trials_checkpoints_can_be_removed(self) -> List[int]:
        """
        See the docstring of the chosen base_scheduler for details
        :return: IDs of paused trials for which checkpoints can be removed
        """
        return self.base_scheduler.trials_checkpoints_can_be_removed()

    def metric_names(self) -> List[str]:
        """
        :return: List of metric names.
        """
        return self.metric

    def metric_mode(self) -> Union[str, List[str]]:
        """
        :return: "min" if target metric is minimized, otherwise "max".
        """
        return self.mode

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata of the scheduler
        """
        return {
            **super(LinearScalarizedScheduler, self).metadata(),
            "scalarized_metric": self.single_objective_metric,
        }

    def is_multiobjective_scheduler(self) -> bool:
        """
        Return True if a scheduler is multi-objective.
        """
        return True
