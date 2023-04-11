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
from typing import Optional, Callable

from syne_tune.callbacks.remove_checkpoints_callback import RemoveCheckpointsCallback
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.remove_checkpoints import (
    RemoveCheckpointsSchedulerMixin,
)
from syne_tune.tuner_callback import TunerCallback
from syne_tune.tuning_status import TuningStatus


def early_checkpoint_removal_factory(
    scheduler: TrialScheduler,
    stop_criterion: Callable[[TuningStatus], bool],
) -> Optional[TunerCallback]:
    """
    Early checkpoint removal is implemented by callbacks, which depend on which
    scheduler is being used. For many schedulers, early checkpoint removal is
    not supported.

    :param scheduler: Scheduler for which early checkpoint removal is requested
    :param stop_criterion: Stop criterion as passed to :class:`~syne_tune.Tuner`
    :return: Callback for early checkpoint removal, or ``None`` if this is not
        supported for the scheduler
    """
    callback = None
    if isinstance(scheduler, RemoveCheckpointsSchedulerMixin):
        callback_kwargs = scheduler.params_early_checkpoint_removal()
    else:
        callback_kwargs = None
    if callback_kwargs is not None:
        # General case: Scheduler implements ``trials_checkpoints_can_be_removed``
        # method, and we can use the generic ``RemoveCheckpointsCallback``
        # callback
        assert len(callback_kwargs) == 0, (
            "params_early_checkpoint_removal of your scheduler returns "
            "arguments, which are not used in RemoveCheckpointsCallback"
        )
        callback = RemoveCheckpointsCallback()
    return callback
