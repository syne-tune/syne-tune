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
from abc import ABC
from time import perf_counter, time
from typing import Dict, Optional, List, Tuple
import copy

from syne_tune.backend.trial_status import Trial

import pandas as pd

from syne_tune.constants import SMT_DECISION, SMT_TRIAL_ID, SMT_STATUS, SMT_TUNER_TIME
from syne_tune.util import RegularCallback


class TunerCallback(ABC):

    def on_tuning_start(self, tuner):
        pass

    def on_tuning_end(self):
        pass

    def on_loop_start(self):
        pass

    def on_loop_end(self):
        pass

    def on_fetch_status_results(
            self,
            trial_status_dict: Tuple[Dict[int, Tuple[Trial, str]]],
            new_results: List[Tuple[int, Dict]],
    ):
        """
        Called with the results of `backend.fetch_status_results`.
        """
        pass

    def on_trial_complete(self, trial: Trial, result: Dict):
        """
        Called when a trial completes.
        :param trial: trial that just completed.
        :param result: last result obtained.
        :return:
        """
        pass

    def on_trial_result(self, trial: Trial, status: str, result: Dict, decision: str):
        """
        Called when a new result is observed.
        :param trial:
        :param status: a string representing the status that is one value of `trial_status.Status`, such as
        `trial_status.Status.completed`.
        :param result:
        :param decision: decision that was returned by the scheduler
        :param scheduler: scheduler object that is passed in case the analysis needs to associate special
        Scheduler information (for instance GP posterior).
        :return:
        """
        pass

    def on_tuning_sleep(self, sleep_time: float):
        """
        Called when the tuner is put to sleep when no worker is available.
        :param sleep_time:
        :return:
        """
        pass


class StoreResultsCallback(TunerCallback):
    def __init__(
            self,
            add_wallclock_time: bool = True,
    ):
        """
        Minimal callback that enables plotting results over time,
        additional callback functionalities will be added as well as example to plot results over time.

        :param add_wallclock_time: whether to add wallclock time to results.
        :param csv_file: if passed results are updated into the csv file, support local and S3 paths. In case an S3 path
        is used `fsspec` and `s3fs` should be installed.
        """
        self.results = []
        self.start = perf_counter() if add_wallclock_time else None

        self.csv_file = None
        self.save_results_at_frequency = None

    def _set_time_fields(self, result: Dict):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the back-end)
        """
        if self.start is not None and SMT_TUNER_TIME not in result:
            result[SMT_TUNER_TIME] = perf_counter() - self.start

    def on_trial_result(self, trial: Trial, status: str, result: Dict, decision: str):
        assert self.save_results_at_frequency is not None, \
            "on_tuning_start must always be called before on_trial_result."
        result = copy.copy(result)
        result[SMT_DECISION] = decision
        result[SMT_STATUS] = status
        result[SMT_TRIAL_ID] = trial.trial_id

        for key in trial.config:
            result[f'config_{key}'] = trial.config[key]

        self._set_time_fields(result)

        self.results.append(result)

        if self.csv_file is not None:
            self.save_results_at_frequency()

    def store_results(self):
        if self.csv_file is not None:
            self.dataframe().to_csv(self.csv_file, index=False)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def on_tuning_start(self, tuner):
        # we set the path of the csv file once the tuner is created since the path may change when the tuner is stop
        # and resumed again on a different machine.
        self.csv_file = str(tuner.tuner_path / "results.csv.zip")

        # we only save results every `results_update_frequency` seconds as this operation
        # may be expensive on remote storage.
        self.save_results_at_frequency = RegularCallback(
             lambda: self.store_results(),
             call_seconds_frequency=tuner.results_update_interval,
        )

    def on_tuning_end(self):
        # store the results in case some results were not commited yet (since they are saved every
        # `results_update_interval` seconds)
        self.store_results()