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
from typing import List, Dict, Optional
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, SchedulerDecision
from syne_tune.optimizer.schedulers import HyperbandScheduler

logger = logging.getLogger(__name__)


class TrialEvent:
    def __init__(self, trial_id: int):
        self.trial_id = trial_id

    def update_scheduler(
            self, trial_scheduler: TrialScheduler, config: Optional[Dict] = None
    ) -> Optional[str]:
        return None


class SuggestEvent(TrialEvent):
    def __init__(self, trial_id: int, config: Dict):
        super(SuggestEvent, self).__init__(trial_id=trial_id)
        self.config = config

    def update_scheduler(
            self, trial_scheduler: TrialScheduler, config: Optional[Dict] = None
    ) -> Optional[str]:
        logger.debug(f"Replaying {self}")
        # Note: Have to make sure (by using `points_to_evaluate`) that config returned here
        # is equal to `self.config`.
        suggestion = trial_scheduler.suggest(trial_id=self.trial_id)
        assert suggestion.spawn_new_trial_id
        assert suggestion.checkpoint_trial_id is None
        trial = Trial(trial_id=self.trial_id, config=self.config, creation_time=None)
        trial_scheduler.on_trial_add(trial=trial)
        return None

    def __repr__(self):
        return f"Suggest(trial_id={self.trial_id}, config={self.config})"


class ResultEvent(TrialEvent):
    def __init__(self, trial_id: int, result: Dict):
        super(ResultEvent, self).__init__(trial_id=trial_id)
        self.result = result

    def update_scheduler(
            self, trial_scheduler: TrialScheduler, config: Optional[Dict] = None
    ) -> Optional[str]:
        logger.debug(f"Replaying {self}")
        trial = Trial(trial_id=self.trial_id, config=config, creation_time=None)
        trial_decision = trial_scheduler.on_trial_result(
            trial=trial, result=self.result)
        return trial_decision

    def __repr__(self):
        return (f"ResultEvent(trial_id={self.trial_id}, result={self.result})")


class ErrorEvent(TrialEvent):
    def __init__(self, trial_id: int):
        super(ErrorEvent, self).__init__(trial_id=trial_id)

    def update_scheduler(
            self, trial_scheduler: TrialScheduler, config: Optional[Dict] = None
    ) -> Optional[str]:
        assert config is not None
        logger.debug(f"Replaying {self}")
        trial = Trial(trial_id=self.trial_id, config=config, creation_time=None)
        trial_scheduler.on_trial_error(trial=trial)
        return None

    def __repr__(self):
        return (
            f"ErrorEvent(trial_id={self.trial_id})")


def replay_scheduling_events(
        events_to_replay: List[TrialEvent],
        config_space: dict,
        **scheduler_kwargs
) -> dict:
    """
    Creates :class:`HyperbandScheduler` and replays events from
    `events_to_replay`. We return the resulting scheduler as well as the
    trial_ids corresponding to :class:`ResultEvent` events resulting in
    STOP or PAUSE.

    Replaying events is a simple and general way to implement a stateless
    API.

    :param events_to_replay:
    :param config_space: Argument for `HyperbandScheduler`
    :param scheduler_kwargs: Arguments for `HyperbandScheduler`
    :return: `HyperbandScheduler` object after replaying
    """
    # Cannot rely on `suggest` to produce the same config's during replay, so
    # we use `points_to_evaluate` to enforce the same sequence
    points_to_evaluate = []
    expected_trial_id = 0
    for event in events_to_replay:
        if isinstance(event, SuggestEvent):
            assert event.trial_id == expected_trial_id
            expected_trial_id += 1
            points_to_evaluate.append(event.config)
    scheduler_kwargs["points_to_evaluate"] = points_to_evaluate
    trial_scheduler = HyperbandScheduler(config_space, **scheduler_kwargs)
    results = []
    for event in events_to_replay:
        results.append(event.update_scheduler(
            trial_scheduler=trial_scheduler,
            config=points_to_evaluate[event.trial_id],
        ))
    # Determine which trial_id's are stopped / paused due to ResultEvent
    # decisions
    stopped_trial_ids = set()
    paused_trial_ids = set()
    for event, result in zip(events_to_replay, results):
        if result == SchedulerDecision.STOP:
            stopped_trial_ids.add(event.trial_id)
        elif result == SchedulerDecision.PAUSE:
            paused_trial_ids.add(event.trial_id)
    return {
        "trial_scheduler": trial_scheduler,
        "stopped_trial_ids": list(stopped_trial_ids),
        "paused_trial_ids": list(paused_trial_ids),
    }
