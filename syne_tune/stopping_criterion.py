import logging
import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict

from syne_tune.tuning_status import TuningStatus

logger = logging.getLogger(__name__)


@dataclass
class StoppingCriterion:
    """
    Stopping criterion that can be used in a Tuner, for instance
    :code:`Tuner(stop_criterion=StoppingCriterion(max_wallclock_time=3600), ...)`.

    If several arguments are used, the combined criterion is true whenever
    one of the atomic criteria is true.

    In principle, ``stop_criterion`` for ``Tuner`` can be any lambda function, but
    this class should be used with remote launching in order to ensure
    proper serialization.

    :param max_wallclock_time: Stop once this wallclock time is reached
    :param max_num_evaluations: Stop once more than this number of metric
        records have been reported
    :param max_num_trials_started: Stop once more than this number of trials
        have been started
    :param max_num_trials_completed: Stop once more than this number of trials
        have been completed. This does not include trials which were stopped
        or failed
    :param max_cost: Stop once total cost of evaluations larger than this value
    :param max_num_trials_finished: Stop once more than this number of trials
        have finished (i.e., completed, stopped, failed, or stopping)
    :param min_metric_value: Dictionary with thresholds for selected metrics.
        Stop once an evaluation reports a metric value below a threshold
    :param max_metric_value: Dictionary with thresholds for selected metrics.
        Stop once an evaluation reports a metric value above a threshold
    """

    max_wallclock_time: float = None
    max_num_evaluations: int = None
    max_num_trials_started: int = None
    max_num_trials_completed: int = None
    max_cost: float = None
    max_num_trials_finished: int = None
    # minimum value for metrics, any value below this threshold will trigger a stop
    min_metric_value: Optional[Dict[str, float]] = None
    # maximum value for metrics, any value above this threshold will trigger a stop
    max_metric_value: Optional[Dict[str, float]] = None

    # todo we should have unit-test for all those cases.
    def __call__(self, status: TuningStatus) -> bool:
        if (
            self.max_wallclock_time is not None
            and status.wallclock_time > self.max_wallclock_time
        ):
            logger.info(
                f"reaching max wallclock time ({self.max_wallclock_time}), stopping there."
            )
            return True
        if (
            self.max_num_trials_started is not None
            and status.num_trials_started > self.max_num_trials_started
        ):
            logger.info(
                f"reaching max number of trials started ({self.max_num_trials_started + 1}), stopping there."
            )
            return True
        if (
            self.max_num_trials_completed is not None
            and status.num_trials_completed > self.max_num_trials_completed
        ):
            logger.info(
                f"reaching max number of trials completed ({self.max_num_trials_completed + 1}), stopping there."
            )
            return True
        if (
            self.max_num_trials_finished is not None
            and status.num_trials_finished > self.max_num_trials_finished
        ):
            logger.info(
                f"reaching max number of trials finished ({self.max_num_trials_finished + 1}), stopping there."
            )
            return True
        if self.max_cost is not None and status.cost > self.max_cost:
            logger.info(f"reaching max cost ({self.max_cost}), stopping there.")
            return True
        if (
            self.max_num_evaluations is not None
            and status.overall_metric_statistics.count > self.max_num_evaluations
        ):
            logger.info(
                f"reaching {status.overall_metric_statistics.count + 1} evaluations, stopping there. "
            )
            return True
        if (
            self.max_metric_value is not None
            and status.overall_metric_statistics.count > 0
        ):
            max_metrics_observed = status.overall_metric_statistics.max_metrics
            for metric, max_metric_accepted in self.max_metric_value.items():
                if (
                    metric in max_metrics_observed
                    and max_metrics_observed[metric] > max_metric_accepted
                ):
                    logger.info(
                        f"found {metric} with value ({max_metrics_observed[metric]}), "
                        f"above the provided threshold {max_metric_accepted} stopping there."
                    )
                    return True

        if (
            self.min_metric_value is not None
            and status.overall_metric_statistics.count > 0
        ):
            min_metrics_observed = status.overall_metric_statistics.min_metrics
            for metric, min_metric_accepted in self.min_metric_value.items():
                if (
                    metric in min_metrics_observed
                    and min_metrics_observed[metric] < min_metric_accepted
                ):
                    logger.info(
                        f"found {metric} with value ({min_metrics_observed[metric]}), "
                        f"below the provided threshold {min_metric_accepted} stopping there."
                    )
                    return True
        return False


class PlateauStopper(object):
    """
    Stops the experiment when a metric plateaued for N consecutive trials
    for more than the given amount of iterations specified in the patience parameter.
    This code is inspired by Ray Tune.
    """

    def __init__(
        self,
        metric: str,
        std: float = 0.001,
        num_trials: int = 10,
        mode: str = "min",
        patience: int = 0,
    ):
        """
        :param metric: The metric to be monitored.
        :param std: The minimal standard deviation after which the tuning process
            has to stop.
        :param num_trials: The number of consecutive trials
        :param mode: The mode to select the top results ("min" or "max")
        :param patience: Number of iterations to wait for a change in the top models.
        """
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be either min or max.")

        if not isinstance(num_trials, int) or num_trials <= 1:
            raise ValueError(
                "Top results to consider must be"
                " a positive integer greater than one."
            )
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be a strictly positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError(
                "The standard deviation must be a strictly positive float number."
            )
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._std = std
        self._num_trials = num_trials

        if self._mode == "min":
            self.multiplier = 1
        else:
            self.multiplier = -1

    def __call__(self, status: TuningStatus) -> bool:
        """Return a boolean representing if the tuning has to stop."""
        if status.num_trials_finished == 0:
            return False

        trials = status.trial_rows
        trajectory = []
        curr_best = None

        for ti in trials.values():
            if self._metric in ti:
                y = self.multiplier * ti[self._metric]
                if curr_best is None or y < curr_best:
                    curr_best = y
                trajectory.append(curr_best)

        top_values = trajectory[-self._num_trials :]
        # If the current iteration has to stop
        has_plateaued = (
            len(top_values) == self._num_trials and np.std(top_values) <= self._std
        )
        if has_plateaued:
            # we increment the total counter of iterations
            self._iterations += 1
        else:
            # otherwise we reset the counter
            self._iterations = 0

        # and then call the method that re-executes
        # the checks, including the iterations.
        return has_plateaued and self._iterations >= self._patience
