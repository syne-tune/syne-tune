import json
import logging
import re
import sys
from dataclasses import dataclass
from time import time, perf_counter
from typing import List, Dict, Any

from syne_tune.constants import (
    ST_WORKER_TIME,
    ST_WORKER_COST,
    ST_WORKER_TIMESTAMP,
    ST_WORKER_ITER,
    ST_METRIC_TAG,
)
from syne_tune.util import dump_json_with_numpy

logging.basicConfig()
logger = logging.getLogger(__name__)


@dataclass
class Reporter:
    """
    Callback for reporting metric values from a training script back to Syne Tune.
    Example:

    .. code-block:: python

       from syne_tune import Reporter

       report = Reporter()
       for epoch in range(1, epochs + 1):
           # ...
           report(epoch=epoch, accuracy=accuracy)

    :param add_time: If True (default), the time (in secs) since creation of the
        :class:`Reporter` object is reported automatically as
        :const:`~syne_tune.constants.ST_WORKER_TIME`
    """

    add_time: bool = True

    def __post_init__(self):
        if self.add_time:
            self.start = perf_counter()
            self.iter = 0

    def __call__(self, **kwargs) -> None:
        """Report metric values from training function back to Syne Tune

        A time stamp :const:`~syne_tune.constants.ST_WORKER_TIMESTAMP` is added.
        See :attr:`add_time` comments.

        :param kwargs: Keyword arguments for metrics to be reported, for instance
            :code:`report(epoch=1, loss=1.2)`. Values must be serializable with json,
            keys should not start with ``st_`` which is a reserved namespace for
            Syne Tune internals.
        """
        self._check_reported_values(kwargs)
        assert not any(key.startswith("st_") for key in kwargs), (
            "The metric prefix 'st_' is used by Syne Tune internals, "
            "please use a metric name that does not start with 'st_'."
        )
        kwargs[ST_WORKER_TIMESTAMP] = time()
        if self.add_time:
            seconds_spent = perf_counter() - self.start
            kwargs[ST_WORKER_TIME] = seconds_spent
            # second cost will only be there if we were able to properly detect the instance-type and instance-count
            # from the environment
            if hasattr(self, "dollar_cost"):
                kwargs[ST_WORKER_COST] = seconds_spent * self.dollar_cost
        kwargs[ST_WORKER_ITER] = self.iter
        self.iter += 1
        _report_logger(**kwargs)

    @staticmethod
    def _check_reported_values(kwargs: Dict[str, Any]):
        assert all(
            v is not None for v in kwargs.values()
        ), f"Invalid value in report: kwargs = {kwargs}"


def _report_logger(**kwargs):
    print(f"[{ST_METRIC_TAG}]: {_serialize_report_dict(kwargs)}")
    sys.stdout.flush()


def _serialize_report_dict(report_dict: Dict[str, Any]) -> str:
    """
    :param report_dict: a dictionary of metrics to be serialized
    :return: serialized string of the reported metrics, an exception is raised if the size is too large or
    if the dictionary values are not JSON-serializable
    """
    try:
        report_str = dump_json_with_numpy(report_dict)
        assert sys.getsizeof(report_str) < 50_000
        return report_str
    except TypeError as e:
        print("The dictionary set to be reported does not seem to be serializable.")
        raise e
    except AssertionError as e:
        print("The dictionary set to be reported is too large.")
        raise e
    except Exception as e:
        raise e


def retrieve(log_lines: List[str]) -> List[Dict[str, float]]:
    """Retrieves metrics reported with :func:`_report_logger` given log lines.

    :param log_lines: Lines in log file to be scanned for metric reports
    :return: list of metrics retrieved from the log lines.
    """
    metrics = []
    regex = r"\[" + ST_METRIC_TAG + r"\]: (\{.*\})"
    for metric_values in re.findall(regex, "\n".join(log_lines)):
        metrics.append(json.loads(metric_values))
    return metrics
