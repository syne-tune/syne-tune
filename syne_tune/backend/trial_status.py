# requires python 3.7
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

try:
    from typing_extensions import Literal
except ImportError as e:
    logging.debug(e)
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
    # Metrics recorded for each call of ``report``. Each metric is a dictionary from metric name to value (
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
