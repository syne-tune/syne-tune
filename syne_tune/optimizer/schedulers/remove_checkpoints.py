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
from typing import List, Optional, Dict, Any


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
    :func:``~syne_tune.callbacks.checkpoint_removal_factory.early_checkpoint_removal_factory`.
    There are different ways how this callback works (they differ in terms
    of which pause-and-resume schedulers use them), all are supported here:

    * General case: Scheduler reports trials for which checkpoints can be
      removed. To this end, the scheduler implements
      :meth:`trials_checkpoints_can_be_removed`. In this case, the callback can
      be very simple:
      :class:`~syne_tune.callbacks.remove_checkpoints_callback.RemoveCheckpointsCallback`.
      For this case, :meth:`params_early_checkpoint_removal` should return an
      empty dictionary.
      Example:
      :class:`~syne_tune.optimizer.scheduler.synchronous.SynchronousHyperbandScheduler`
    * Special cases: The callback implements a more advanced logic for a specific
      kind of scheduler. The callback is created by
      ``early_checkpoint_removal_factory``, and
      :meth:`params_early_checkpoint_removal` provides the constructor arguments.
      Example:
      :class:`~syne_tune.callbacks.hyperband_remove_checkpoints_callback.HyperbandRemoveCheckpointsCallback`.
      Note that in this case, the scheduler may need to provide information to
      the callback via methods not specified here.
    """

    def trials_checkpoints_can_be_removed(self) -> List[int]:
        """
        Supports the general case (see header comment).
        This method returns IDs of paused trials for which checkpoints can safely
        be removed. These trials either cannot be resumed anymore, or it is very
        unlikely they will be resumed. Any trial ID needs to be returned only once,
        not over and over. If a trial gets stopped (by returning
        :const:`SchedulerDecision.STOP` in :meth:`on_trial_result`), its checkpoint
        is removed anyway, so its ID does not have to be returned here.

        :return: IDs of paused trials for which checkpoints can be removed
        """
        return []  # Safe default

    def params_early_checkpoint_removal(self) -> Optional[Dict[str, Any]]:
        """
        Supports special cases, in which :meth:`trials_checkpoints_can_be_removed`
        is not used. In such cases, the checkpoint removal callback is created by
        :func:``~syne_tune.callbacks.checkpoint_removal_factory.early_checkpoint_removal_factory`.
        The arguments for the callback constructor are provided here.

        .. note::
           In the general case, where :meth:`trials_checkpoints_can_be_removed`
           is implemented, this method should return an empty dictionary.
           If this method returns ``None``, it means that early checkpoint removal
           should not be done.

        :return: Arguments for callback constructor, or ``None`` if early
            checkpoint removal is not supported
        """
        return None
