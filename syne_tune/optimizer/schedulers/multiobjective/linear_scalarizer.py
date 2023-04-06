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
from collections.abc import Iterable
from itertools import groupby
from typing import Dict, Any, List, Union

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.fifo import _to_list

logger = logging.getLogger(__name__)


def _all_equal(iterable: Iterable) -> bool:
    """
    Check if all elements of an iterable are the same
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class LinearScalarizedFIFOScheduler(FIFOScheduler):
    """Scheduler with a Linear Scalarization of multiple objectives.
    This method optimizes a single objective equal to the linear scalarization of given two objectives.
    The scalarized single objective is named: 'scalarized_<metric1>_<metric2>_..._<metricN>'

    See :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Names of metrics to optimize
    :param mode: Modes of metrics to optimize (min or max). All must be matching.
    :param scalarization_weights: Weights used to scalarize objectives, if None an array of 1s is used
    :param searcher_name: Name of the Single objective Searcher used for optimizing the linearized objective
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    scalarization_weights: np.ndarray
    multi_objective_metrics: List[str]

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        scalarization_weights: Union[np.ndarray, List[float]] = None,
        mode: Union[List[str], str] = "min",
        searcher="bayesopt",
        **kwargs,
    ):
        self.multi_objective_metrics = metric
        if scalarization_weights is None:
            scalarization_weights = np.ones(shape=len(metric))
        self.scalarization_weights = np.asarray(scalarization_weights)

        assert len(mode) >= 1, "At least one mode must be provided"
        if isinstance(mode, Iterable):
            assert _all_equal(
                mode
            ), "Modes must be the same, use positive/negative scalarization_weights to change relative signs"
            mode = next(x for x in mode)

        metric = f"scalarized_{'_'.join(metric)}"
        super(LinearScalarizedFIFOScheduler, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher,
            mode=mode,
            **kwargs,
        )

    def multi_objective_metric_names(self) -> List[str]:
        return _to_list(self.multi_objective_metrics)

    def _multi_objective_check_result(self, result: Dict[str, Any]):
        self._check_keys_of_result(result, self.multi_objective_metric_names())

    def _scalarized_metric(self, result: Dict[str, Any]) -> float:
        self._multi_objective_check_result(result)
        mo_results = np.array(
            [result[item] for item in self.multi_objective_metric_names()]
        )
        logger.debug(
            f"Scalarizing {self.multi_objective_metric_names()} into {self.metric} using linear combination"
        )
        return np.sum(np.multiply(mo_results, self.scalarization_weights))

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """
        We simply relay ``result`` to the searcher. Other decisions are done
        in ``on_trial_complete``.
        """
        result[self.metric] = self._scalarized_metric(result)
        return super(LinearScalarizedFIFOScheduler, self).on_trial_result(trial, result)

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        result[self.metric] = self._scalarized_metric(result)
        return super(LinearScalarizedFIFOScheduler, self).on_trial_complete(
            trial, result
        )

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata of the scheduler
        """
        return {
            **super(LinearScalarizedFIFOScheduler, self).metadata(),
            "multi_objective_metric": self.multi_objective_metric_names(),
        }
