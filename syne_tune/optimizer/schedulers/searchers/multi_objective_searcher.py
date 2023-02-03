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
from typing import Optional, List

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher, impute_points_to_evaluate


logger = logging.getLogger(__name__)


class BaseMultiObjectiveSearcher(BaseSearcher):
    """
    Base class for searchers that optimize multiple objectives simultaneously.
    Derivatives of this class need to implement :meth:`~get_config`.

    :param config_space: Configuration space
    :param metrics: Name of all metrics that should be optimized.
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :param modes: Defines for each metric if it should be minimized ("min", default) or maximized
        ("max")
    """

    def __init__(
        self,
        config_space: dict,
        metrics: List[str],
        modes: Optional[List[dict], str] = 'min',
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        self.config_space = config_space
        self._metrics = metrics
        self._points_to_evaluate = impute_points_to_evaluate(
            points_to_evaluate, config_space
        )
        self._modes = modes

    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        :param scheduler: Scheduler the searcher is used with.
        :type scheduler: :class:`~syne_tune.optimizer.schedulers.TrialScheduler`
        """
        if hasattr(scheduler, "metric"):
            self._metrics = getattr(scheduler, "metrics")
        if hasattr(scheduler, "mode"):
            self._modes = getattr(scheduler, "modes")


