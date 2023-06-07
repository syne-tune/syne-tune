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

from syne_tune.tuner_callback import TunerCallback
from syne_tune.tuning_status import TuningStatus


class RemoveCheckpointsSchedulerMixin:
    """
    Methods to be implemented by pause-and-resume schedulers (in that
    :meth:`on_trial_result` can return :const:`SchedulerDecision.PAUSE`) which
    support early removal of checkpoints. Typically, model checkpoints are
    retained for paused trials, because they may get resumed later on. This can
    lead to the disk filling up, so removing checkpoints which are no longer
    needed, can be important.

    Early checkpoint removal is implemented as a callback used with
    :class:`~syne_tune.Tuner`, which is created by
    :meth:`callback_for_checkpoint_removal` here.
    """

    def callback_for_checkpoint_removal(
        self, stop_criterion: Callable[[TuningStatus], bool]
    ) -> Optional[TunerCallback]:
        """
        :param stop_criterion: Stopping criterion, as passed to
            :class:`~syne_tune.Tuner`
        :return: CP removal callback, or ``None`` if CP removal is not activated
        """
        raise NotImplementedError
