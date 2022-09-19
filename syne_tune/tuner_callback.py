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
import numbers
import os
from time import perf_counter
from typing import Dict, List, Tuple, Optional
import copy
import logging
import pandas as pd

from syne_tune.backend.trial_status import Trial
from syne_tune.constants import ST_DECISION, ST_TRIAL_ID, ST_STATUS, ST_TUNER_TIME
from syne_tune.util import RegularCallback

logger = logging.getLogger(__name__)


class TunerCallback:
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
        new_results: List[Tuple[int, dict]],
    ):
        """
        Called with the results of `backend.fetch_status_results`.
        """
        pass

    def on_trial_complete(self, trial: Trial, result: dict):
        """
        Called when a trial completes.
        :param trial: trial that just completed.
        :param result: last result obtained.
        :return:
        """
        pass

    def on_trial_result(self, trial: Trial, status: str, result: dict, decision: str):
        """
        Called when a new result is observed.
        :param trial:
        :param status: a string representing the status that is one value of `trial_status.Status`, such as
        `trial_status.Status.completed`.
        :param result:
        :param decision: decision that was returned by the scheduler
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
        """
        self.results = []

        self.csv_file = None
        self.save_results_at_frequency = None
        self.add_wallclock_time = add_wallclock_time
        self._start_time_stamp = None

    def _set_time_fields(self, result: dict):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the back-end)
        """
        if self._start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self._start_time_stamp

    def on_trial_result(self, trial: Trial, status: str, result: dict, decision: str):
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
        if self.add_wallclock_time:
            self._start_time_stamp = perf_counter()

    def on_tuning_end(self):
        # store the results in case some results were not committed yet (since they are saved every
        # `results_update_interval` seconds)
        self.store_results()


class TensorboardCallback(TunerCallback):
    def __init__(
        self,
        ignore_metrics: Optional[List[str]] = None,
        target_metric: Optional[str] = None,
        mode: Optional[str] = "min",
    ):
        """
        Simple callback that logs metric reported in the train function such that we can visualize with Tensorboard.

        :param ignore_metrics: Defines which metrics should be ignored. If None, all metrics are reported
         to Tensorboard.
        :param target_metric: Defines the metric we aim to optimize. If this argument is set, we report
        the cumulative optimum of this metric as well as the optimal hyperparameters we have found so far.
        :param mode: Determined whether we maximize ('max') or minimize ('min') the target metric.
        """
        self.results = []

        if ignore_metrics is None:
            self.ignore_metrics = []
        else:
            self.ignore_metrics = ignore_metrics

        self.curr_best_value = None
        self.curr_best_config = None

        self.start_time_stamp = None
        self.writer = None
        self.iter = None
        self.mode = mode
        self.target_metric = target_metric
        self.trial_ids = set()

        self.metric_sign = -1 if mode == "max" else 1

    def _set_time_fields(self, result: dict):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the back-end)
        """
        if self.start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self.start_time_stamp

    def on_trial_result(self, trial: Trial, status: str, result: dict, decision: str):
        walltime = result[ST_TUNER_TIME]
        self._set_time_fields(result)

        if self.target_metric is not None:

            assert (
                self.target_metric in result
            ), f"{self.target_metric} was not reported back to Syne tune"
            new_result = self.metric_sign * result[self.target_metric]

            if self.curr_best_value is None or self.curr_best_value > new_result:
                self.curr_best_value = new_result
                self.curr_best_config = trial.config
                self.writer.add_scalar(
                    self.target_metric, result[self.target_metric], self.iter, walltime
                )

            else:
                opt = self.metric_sign * self.curr_best_value
                self.writer.add_scalar(self.target_metric, opt, self.iter, walltime)

            for key, value in self.curr_best_config.items():
                if isinstance(value, numbers.Number):
                    self.writer.add_scalar(f"optimal_{key}", value, self.iter, walltime)
                else:
                    self.writer.add_text(
                        f"optimal_{key}", str(value), self.iter, walltime
                    )

        for metric in result:
            if metric not in self.ignore_metrics:
                self.writer.add_scalar(metric, result[metric], self.iter, walltime)

        for key, value in trial.config.items():
            if isinstance(value, numbers.Number):
                self.writer.add_scalar(key, value, self.iter, walltime)
            else:
                self.writer.add_text(key, str(value), self.iter, walltime)

        self.writer.add_scalar("runtime", result[ST_TUNER_TIME], self.iter, walltime)

        self.trial_ids.add(trial.trial_id)
        self.writer.add_scalar(
            "number_of_trials",
            len(self.trial_ids),
            self.iter,
            walltime=walltime,
            display_name="total number of trials",
        )

        self.iter += 1

    def _create_summary_writer(self):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            logger.error(
                "TensoboardX is not installed. You can install it via: pip install tensorboardX"
            )
            raise
        return SummaryWriter(self.output_path)

    def on_tuning_start(self, tuner):
        self.output_path = os.path.join(tuner.tuner_path, "tensorboard_output")
        self.writer = self._create_summary_writer()
        self.iter = 0
        self.start_time_stamp = perf_counter()
        logger.info(
            f"Logging tensorboard information at {self.output_path}, to visualize results, run\n"
            f"tensorboard --logdir {self.output_path}"
        )

    def on_tuning_end(self):
        self.writer.close()
        logger.info(
            f"Tensorboard information has been logged at {self.output_path}, to visualize results, run\n"
            f"tensorboard --logdir {self.output_path}"
        )

    def __getstate__(self):
        state = {
            "results": self.results,
            "ignore_metrics": self.ignore_metrics,
            "curr_best_value": self.curr_best_value,
            "curr_best_config": self.curr_best_config,
            "start_time_stamp": self.start_time_stamp,
            "iter": self.iter,
            "mode": self.mode,
            "target_metric": self.target_metric,
            "trial_ids": self.trial_ids,
            "metric_sign": self.metric_sign,
            "output_path": self.output_path,
        }
        return state

    def __setstate__(self, state):
        super().__init__(
            ignore_metrics=state["ignore_metrics"],
            target_metric=state["target_metric"],
            mode=state["mode"],
        )
        self.results = state["results"]
        self.curr_best_value = state["curr_best_value"]
        self.curr_best_config = state["curr_best_config"]
        self.start_time_stamp = state["start_time_stamp"]
        self.iter = state["iter"]

        self.trial_ids = state["trial_ids"]
        self.metric_sign = state["metric_sign"]
        self.output_path = state["output_path"]
        self.writer = self._create_summary_writer()
