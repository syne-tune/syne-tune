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
from syne_tune.tuner_callback import TunerCallback
from syne_tune.optimizer.schedulers.remove_checkpoints import (
    RemoveCheckpointsSchedulerMixin,
)


class RemoveCheckpointsCallback(TunerCallback):
    """
    This implements early removal of checkpoints of paused trials. In order
    for this to work, the scheduler needs to implement
    :meth:`~syne_tune.optimizer.scheduler.TrialScheduler.trials_checkpoints_can_be_removed`.
    """

    def __init__(self):
        self._tuner = None

    def on_tuning_start(self, tuner):
        assert isinstance(
            tuner.scheduler, RemoveCheckpointsSchedulerMixin
        ), "tuner.scheduler must be of type RemoveCheckpointsSchedulerMixin"
        self._tuner = tuner

    def on_loop_end(self):
        for trial_id in self._tuner.scheduler.trials_checkpoints_can_be_removed():
            self._tuner.trial_backend.delete_checkpoint(trial_id)
