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
from typing import Optional
import logging

from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.scheduler import TrialSuggestion
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.hb_replay.replay_scheduler import (
    SuggestEvent,
    ResultEvent,
    ErrorEvent,
)

logger = logging.getLogger(__name__)


class HyperbandSchedulerWrapper(HyperbandScheduler):
    r"""
    Version of :class:`HyperbandScheduler` which records events relevant for
    replaying. This is useful for testing the replaying code.
    """

    def __init__(self, config_space, **kwargs):
        scheduler_type = kwargs.get("type", "stopping")
        assert scheduler_type == "stopping", (
            "Replaying only supported for stopping type"
        )
        super().__init__(config_space, **kwargs)
        self.events_to_replay = []

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        suggestion = super().suggest(trial_id)
        if suggestion is not None:
            event = SuggestEvent(trial_id=trial_id, config=suggestion.config)
            self.events_to_replay.append(event)
            logger.debug(f"Recording {event}")
        return suggestion

    def on_trial_error(self, trial: Trial):
        super().on_trial_error(trial)
        event = ErrorEvent(trial_id=trial.trial_id)
        self.events_to_replay.append(event)
        logger.debug(f"Recording {event}")

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        trial_decision = super().on_trial_result(trial=trial, result=result)
        event = ResultEvent(trial_id=trial.trial_id, result=result)
        self.events_to_replay.append(event)
        logger.debug(f"Recording {event}")
        return trial_decision
