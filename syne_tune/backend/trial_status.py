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
# requires python 3.7
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

try:
    from typing_extensions import Literal
except ImportError:
    from typing import Literal


from syne_tune.constants import ST_WORKER_COST


class Status:
    completed: str = "Completed"
    in_progress: str = "InProgress"
    failed: str = "Failed"
    paused: str = "Paused"
    stopped: str = "Stopped"
    stopping: str = "Stopping"


@dataclass
class Trial:
    trial_id: int
    config: Dict[str, object]
    creation_time: datetime

    def add_results(self, metrics, status, training_end_time):
        return TrialResult(
            metrics=metrics,
            status=status,
            training_end_time=training_end_time,
            trial_id=self.trial_id,
            config=self.config,
            creation_time=self.creation_time,
        )


@dataclass
class TrialResult(Trial):
    # Metrics recorded for each call of `report`. Each metric is a dictionary from metric name to value (
    # could be numeric or string, the only constrain is that it must be compatible with json).
    metrics: List[Dict[str, object]]
    status: Literal[
        Status.completed,
        Status.in_progress,
        Status.failed,
        Status.stopped,
        Status.stopping,
    ]

    training_end_time: Optional[datetime] = None

    @property
    def seconds(self):
        # todo the robustness of this logic could be improved. Currently, if the job is still running the runtime is the
        #  difference between training end and start time. If the job is still running, it is the difference between now
        #  and start time. However, it should be the difference between the last updated time and start time.
        end_time = (
            datetime.now() if self.training_end_time is None else self.training_end_time
        )
        return (
            end_time.replace(tzinfo=None) - self.creation_time.replace(tzinfo=None)
        ).seconds

    @property
    def cost(self):
        if len(self.metrics) > 0:
            return self.metrics[-1].get(ST_WORKER_COST, None)
        else:
            return None
