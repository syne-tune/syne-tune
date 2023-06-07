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
from typing import Dict, Any

from syne_tune.backend.trial_status import Trial
from syne_tune.backend.trial_backend import (
    TrialAndStatusInformation,
    TrialIdAndResultList,
)


class TunerCallback:
    """
    Allows user of :class:`~syne_tune.Tuner` to monitor progress, store
    additional results, etc.
    """

    def on_tuning_start(self, tuner):
        """Called at start of tuning loop

        :param tuner: :class:`~syne_tune.Tuner` object
        """
        pass

    def on_tuning_end(self):
        """Called once the tuning loop terminates

        This is called before :class:`~syne_tune.Tuner` object is serialized
        (optionally), and also before running jobs are stopped.
        """
        pass

    def on_loop_start(self):
        """Called at start of each tuning loop iteration

        Every iteration starts with fetching new results from the backend.
        This is called before this is done.
        """
        pass

    def on_loop_end(self):
        """Called at end of each tuning loop iteration

        This is done before the loop stopping condition is checked and acted
        upon.
        """
        pass

    def on_fetch_status_results(
        self,
        trial_status_dict: TrialAndStatusInformation,
        new_results: TrialIdAndResultList,
    ):
        """Called just after ``trial_backend.fetch_status_results``

        :param trial_status_dict: Result of ``fetch_status_results``
        :param new_results: Result of ``fetch_status_results``
        """
        pass

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Called when a trial completes (``Status.completed``)

        The arguments here also have been passed to ``scheduler.on_trial_complete``,
        before this call here.

        :param trial: Trial that just completed.
        :param result: Last result obtained.
        """
        pass

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        """Called when a new result (reported by a trial) is observed

        The arguments here are inputs or outputs of ``scheduler.on_trial_result``
        (called just before).

        :param trial: Trial whose report has been received
        :param status: Status of trial before ``scheduler.on_trial_result`` has
            been called
        :param result: Result dict received
        :param decision: Decision returned by ``scheduler.on_trial_result``
        """
        pass

    def on_tuning_sleep(self, sleep_time: float):
        """Called just after tuner has slept, because no worker was available

        :param sleep_time: Time (in secs) for which tuner has just slept
        """
        pass

    def on_start_trial(self, trial: Trial):
        """Called just after a new trials is started

        :param trial: Trial which has just been started
        """
        pass

    def on_resume_trial(self, trial: Trial):
        """Called just after a trial is resumed

        :param trial: Trial which has just been resumed
        """
        pass
