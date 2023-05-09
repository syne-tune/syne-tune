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
from typing import Optional, List, Dict, Any
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.tuner_callback import TunerCallback

logger = logging.getLogger(__name__)


class TensorboardCallback(TunerCallback):
    """
    Logs relevant metrics reported from trial evaluations, so they can be
    visualized with Tensorboard.

    :param ignore_metrics: Defines which metrics should be ignored. If None,
        all metrics are reported to Tensorboard.
    :param target_metric: Defines the metric we aim to optimize. If this
        argument is set, we report the cumulative optimum of this metric as
        well as the optimal hyperparameters we have found so far.
    :param mode: Determined whether we maximize ("max") or minimize ("min")
        the target metric.
    :param log_hyperparameters: If set to True, we also log all hyperparameters
        specified in the configurations space.
    """

    def __init__(
        self,
        ignore_metrics: Optional[List[str]] = None,
        target_metric: Optional[str] = None,
        mode: Optional[str] = None,
        log_hyperparameters: bool = True,
    ):
        if mode is None:
            mode = "min"
        else:
            assert mode in {
                "min",
                "max",
            }, f"mode = '{mode}' is invalid, use 'max' or 'min'"
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
        self.target_metric = target_metric
        self.trial_ids = set()
        self.metric_sign = -1 if mode == "max" else 1
        self.output_path = None
        self.log_hyperparameters = log_hyperparameters

    def _set_time_fields(self, result: Dict[str, Any]):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the backend)
        """
        if self.start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self.start_time_stamp

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        self._set_time_fields(result)
        walltime = result[ST_TUNER_TIME]

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

        if self.log_hyperparameters:
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
        except ImportError as err:
            print(
                "TensorboardCallback requires tensorboardX to be installed:\n"
                "   pip install tensorboardX\n\n" + str(err)
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
        state = self.__dict__.copy()
        del state["writer"]
        return state

    def __setstate__(self, state):
        self.__init__(
            ignore_metrics=state["ignore_metrics"],
            target_metric=state["target_metric"],
            mode="min" if state["metric_sign"] == 1 else "max",
            log_hyperparameters=state["log_hyperparameters"],
        )
        self.results = state["results"]
        self.curr_best_value = state["curr_best_value"]
        self.curr_best_config = state["curr_best_config"]
        self.start_time_stamp = state["start_time_stamp"]
        self.iter = state["iter"]

        self.trial_ids = state["trial_ids"]
        self.output_path = state["output_path"]
        self.writer = self._create_summary_writer()
