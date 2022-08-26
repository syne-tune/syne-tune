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
import logging
import time
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple
import pandas as pd
from numpy import inf as np_inf

from syne_tune.backend.trial_status import Status, Trial
from syne_tune.constants import ST_WORKER_TIME, ST_WORKER_COST


class MetricsStatistics:
    def __init__(self):
        """
        Allows to maintain simple running statistics (min/max/sum/count) of metrics provided.
        Statistics are tracked for numeric types only. Types of first added metrics define its types.
        """
        self.metric_names = []
        self.count = 0
        self.min_metrics = {}
        self.max_metrics = {}
        self.sum_metrics = {}
        self.last_metrics = {}
        self.is_numeric = {}

    def add(self, metrics: Dict):
        for metric_name, current_metric in metrics.items():
            if metric_name in self.is_numeric:
                if self.is_numeric[metric_name] != isinstance(
                    current_metric, numbers.Number
                ):
                    logging.warning(
                        f"Numeric and non-numeric values reported for metric {metric_name}."
                    )
            if self.is_numeric.get(metric_name, True):
                self.is_numeric[metric_name] = isinstance(
                    current_metric, numbers.Number
                )
                if self.is_numeric[metric_name]:
                    self.min_metrics[metric_name] = min(
                        self.min_metrics.get(metric_name, np_inf), current_metric
                    )
                    self.max_metrics[metric_name] = max(
                        self.max_metrics.get(metric_name, -np_inf), current_metric
                    )
                    self.sum_metrics[metric_name] = (
                        self.sum_metrics.get(metric_name, 0) + current_metric
                    )
        self.metric_names = list(self.min_metrics.keys())
        self.last_metrics = metrics
        self.count += 1


class TuningStatus:
    """
    Information of a tuning job to display as progress or to use to decide whether to stop the tuning job.
    """

    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.start_time = time.perf_counter()

        self.overall_metric_statistics = MetricsStatistics()
        self.trial_metric_statistics = defaultdict(lambda: MetricsStatistics())

        self.last_trial_status_seen = OrderedDict()
        self.trial_rows = OrderedDict({})

    def update(
        self,
        trial_status_dict: Dict[int, Tuple[Trial, str]],
        new_results: List[Tuple[int, Dict]],
    ):
        """
        Updates the tuning status given new statuses and results.
        """

        self.last_trial_status_seen.update(
            {k: v[1] for k, v in trial_status_dict.items()}
        )

        for trial_id, new_result in new_results:
            self.overall_metric_statistics.add(new_result)
            self.trial_metric_statistics[trial_id].add(new_result)

        for trial_id, (trial, status) in trial_status_dict.items():
            num_metrics = self.trial_metric_statistics[trial_id].count
            row = {
                "trial_id": trial_id,
                "status": status,
                "iter": num_metrics,
            }
            row.update(trial.config)
            row.update(self.trial_metric_statistics[trial_id].last_metrics)

            if ST_WORKER_TIME in self.trial_metric_statistics[trial_id].max_metrics:
                row["worker-time"] = self.trial_metric_statistics[trial_id].max_metrics[
                    ST_WORKER_TIME
                ]
            if ST_WORKER_COST in self.trial_metric_statistics[trial_id].max_metrics:
                row["worker-cost"] = self.trial_metric_statistics[trial_id].max_metrics[
                    ST_WORKER_COST
                ]

            self.trial_rows[trial_id] = row

    def mark_running_job_as_stopped(self):
        """
        Update the status of all trials still running to be marked as stop.
        """
        self.last_trial_status_seen = {
            k: v if v != Status.in_progress else Status.stopped
            for k, v in self.last_trial_status_seen.items()
        }
        for trial_id, row in self.trial_rows.items():
            if row["status"] == Status.in_progress:
                row["status"] = Status.stopped

    @property
    def num_trials_started(self):
        return len(self.last_trial_status_seen)

    def _num_trials(self, status: str):
        return sum(
            trial_status == status
            for trial_status in self.last_trial_status_seen.values()
        )

    @property
    def num_trials_completed(self):
        return self._num_trials(status=Status.completed)

    @property
    def num_trials_failed(self):
        return self._num_trials(status=Status.failed)

    @property
    def num_trials_finished(self):
        """
        :return: number of trials that finished, e.g. that completed, were stopped or are stopping, or failed
        """
        # note it may be inefficient to query several times the dataframe in case a very large number of jobs are
        #  present, we could query the dataframe only once
        return (
            self._num_trials(status=Status.completed)
            + self._num_trials(status=Status.stopped)
            + self._num_trials(status=Status.stopping)
            + self._num_trials(status=Status.failed)
        )

    @property
    def num_trials_running(self):
        return self._num_trials(status=Status.in_progress)

    @property
    def wallclock_time(self):
        """
        :return: the wallclock time spent in the tuner
        """
        return time.perf_counter() - self.start_time

    @property
    def user_time(self):
        """
        :return: the total user time spent in the workers
        """
        if ST_WORKER_TIME in self.overall_metric_statistics.metric_names:
            usertime_per_trial = [
                metric.max_metrics.get(ST_WORKER_TIME, 0)
                for trial, metric in self.trial_metric_statistics.items()
            ]
            return sum(usertime_per_trial)
        else:
            return 0

    @property
    def cost(self):
        """
        :return: the estimated dollar-cost spent while tuning
        """
        if ST_WORKER_COST in self.overall_metric_statistics.metric_names:
            cost_per_trial = [
                metric.max_metrics.get(ST_WORKER_COST, 0)
                for trial, metric in self.trial_metric_statistics.items()
            ]
            return sum(cost_per_trial)
        else:
            return 0.0

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trial_rows.values())

    def __str__(self):
        num_running = self.num_trials_running
        num_finished = self.num_trials_started - num_running

        if len(self.trial_rows) > 0:
            df = self.get_dataframe()
            cols = [col for col in df.columns if not col.startswith("st_")]
            res_str = df.loc[:, cols].to_string(index=False, na_rep="-") + "\n"
        else:
            res_str = ""
        res_str += (
            f"{num_running} trials running, "
            f"{num_finished} finished ({self.num_trials_completed} until the end), "
            f"{self.wallclock_time:.2f}s wallclock-time"
        )
        # f"{self.user_time:.2f}s approximated user-time"
        cost = self.cost
        if cost is not None and cost > 0.0:
            res_str += f", ${cost:.2f} estimated cost"
        res_str += "\n"
        return res_str


def print_best_metric_found(
    tuning_status: TuningStatus, metric_names: List[str], mode: str
) -> Tuple[int, float]:
    """
    Prints trial status summary and the best metric found.
    :param tuning_status:
    :param metric_names:
    :param mode:
    :return: trial-id and value of the best metric found
    """
    if tuning_status.overall_metric_statistics.count == 0:
        return
    # only plot results of the best first metric for now in summary, plotting the optimal metrics for multiple
    # objectives would require to display the Pareto set.
    metric_name = metric_names[0]
    print("-" * 20)
    print(f"Resource summary (last result is reported):\n{str(tuning_status)}")
    if mode == "min":
        metric_per_trial = [
            (trial_id, stats.min_metrics.get(metric_name, np_inf))
            for trial_id, stats in tuning_status.trial_metric_statistics.items()
        ]
        metric_per_trial = sorted(metric_per_trial, key=lambda x: x[1])
    else:
        metric_per_trial = [
            (trial_id, stats.max_metrics.get(metric_name, -np_inf))
            for trial_id, stats in tuning_status.trial_metric_statistics.items()
        ]
        metric_per_trial = sorted(metric_per_trial, key=lambda x: -x[1])
    best_trialid, best_metric = metric_per_trial[0]
    print(f"{metric_name}: best {best_metric} for trial-id {best_trialid}")
    print("-" * 20)
    return best_trialid, best_metric
