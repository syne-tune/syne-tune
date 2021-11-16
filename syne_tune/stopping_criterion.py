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
from dataclasses import dataclass
from typing import Optional, Dict

from syne_tune.tuning_status import TuningStatus

logger = logging.getLogger(__name__)


@dataclass
class StoppingCriterion:
    """
    Stopping criterion that can be used in a Tuner, for instance
    `Tuner(stop_criterion=StoppingCriterion(max_wallclock_time=3600), ...)`
    While a lambda can be used for the Tuner, e.g.
    `Tuner(stop_criterion=lambda status: status.wallclock_time > 3600, ...)`
    Using this class is needed when using the remote launcher to ensure serialization works correctly.
    """
    max_wallclock_time: float = None
    max_num_trials_started: int = None
    max_num_trials_completed: int = None
    max_cost: float = None
    max_num_trials_finished: int = None

    # minimum value for metrics, any value bellow this threshold will trigger a stop
    min_metric_value: Optional[Dict[str, float]] = None
    # maximum value for metrics, any value above this threshold will trigger a stop
    max_metric_value: Optional[Dict[str, float]] = None

    # todo we should have unit-test for all those cases.
    def __call__(self, status: TuningStatus) -> bool:
        if self.max_wallclock_time is not None and status.wallclock_time > self.max_wallclock_time:
            logger.info(f"reaching max wallclock time ({self.max_wallclock_time}), stopping there.")
            return True
        if self.max_num_trials_started is not None and status.num_trials_started > self.max_num_trials_started:
            logger.info(f"reaching max number of trials started ({self.max_num_trials_started}), stopping there.")
            return True
        if self.max_num_trials_completed is not None and status.num_trials_completed > self.max_num_trials_completed:
            logger.info(f"reaching max number of trials completed ({self.max_num_trials_completed}), stopping there.")
            return True
        if self.max_num_trials_finished is not None and status.num_trials_finished > self.max_num_trials_finished:
            logger.info(f"reaching max number of trials finished ({self.max_num_trials_finished}), stopping there.")
            return True
        if self.max_cost is not None and status.cost > self.max_cost:
            logger.info(f"reaching max cost ({self.max_cost}), stopping there.")
            return True

        if self.max_metric_value is not None and status.overall_metric_statistics.count > 0:
            max_metrics_observed = status.overall_metric_statistics.max_metrics
            for metric, max_metric_accepted in self.max_metric_value.items():
                if metric in max_metrics_observed and max_metrics_observed[metric] > max_metric_accepted:
                    logger.info(f"found {metric} with value ({max_metrics_observed[metric]}), "
                                f"above the provided threshold {max_metric_accepted} stopping there.")
                    return True

        if self.min_metric_value is not None and status.overall_metric_statistics.count > 0:
            min_metrics_observed = status.overall_metric_statistics.min_metrics
            for metric, min_metric_accepted in self.min_metric_value.items():
                if metric in min_metrics_observed and min_metrics_observed[metric] < min_metric_accepted:
                    logger.info(f"found {metric} with value ({min_metrics_observed[metric]}), "
                                f"bellow the provided threshold {min_metric_accepted} stopping there.")

                    return True
        return False
