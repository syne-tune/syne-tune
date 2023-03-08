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
from time import perf_counter
import copy
import pandas as pd

from syne_tune.backend.trial_status import Trial
from syne_tune.backend.trial_backend import (
    TrialAndStatusInformation,
    TrialIdAndResultList,
)
from syne_tune.constants import ST_DECISION, ST_TRIAL_ID, ST_STATUS, ST_TUNER_TIME
from syne_tune.util import RegularCallback


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


class StoreResultsCallback(TunerCallback):
    """
    Default implementation of :class:`~TunerCallback` which records all
    reported results, and allows to store them as CSV file.

    :param add_wallclock_time: If True, wallclock time since call of
        ``on_tuning_start`` is stored as
        :const:`~syne_tune.constants.ST_TUNER_TIME`.
    """

    def __init__(
        self,
        add_wallclock_time: bool = True,
    ):
        self.results = []
        self.csv_file = None
        self.save_results_at_frequency = None
        self.add_wallclock_time = add_wallclock_time
        self._start_time_stamp = None

    def _set_time_fields(self, result: Dict[str, Any]):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the backend)
        """
        if self._start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self._start_time_stamp

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        assert (
            self.save_results_at_frequency is not None
        ), "on_tuning_start must always be called before on_trial_result."
        result = copy.copy(result)
        result[ST_DECISION] = decision
        result[ST_STATUS] = status
        result[ST_TRIAL_ID] = trial.trial_id

        for key in trial.config:
            result[f"config_{key}"] = trial.config[key]

        self._set_time_fields(result)

        self.results.append(result)

        if self.csv_file is not None:
            self.save_results_at_frequency()

    def store_results(self):
        """
        Store current results into CSV file, of name
        ``{tuner.tuner_path}/results.csv.zip``.
        """
        if self.csv_file is not None:
            self.dataframe().to_csv(self.csv_file, index=False)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def on_tuning_start(self, tuner):
        # we set the path of the csv file once the tuner is created since the path may change when the tuner is stop
        # and resumed again on a different machine.
        self.csv_file = str(tuner.tuner_path / "results.csv.zip")

        # we only save results every ``results_update_frequency`` seconds as this operation
        # may be expensive on remote storage.
        self.save_results_at_frequency = RegularCallback(
            lambda: self.store_results(),
            call_seconds_frequency=tuner.results_update_interval,
        )
        if self.add_wallclock_time:
            self._start_time_stamp = perf_counter()

    def on_tuning_end(self):
        # store the results in case some results were not committed yet (since they are saved every
        # ``results_update_interval`` seconds)
        self.store_results()
