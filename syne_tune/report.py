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
import os
import re
import sys
import json
import logging
from ast import literal_eval
from typing import List, Dict, Any
from time import time, perf_counter
from dataclasses import dataclass

from syne_tune.constants import (
    ST_INSTANCE_TYPE,
    ST_INSTANCE_COUNT,
    ST_WORKER_TIME,
    ST_WORKER_COST,
    ST_WORKER_TIMESTAMP,
    ST_WORKER_ITER,
    ST_SAGEMAKER_METRIC_TAG,
)
from syne_tune.util import dump_json_with_numpy

# this is required so that metrics are written
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos

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
    :param add_cost: If True (default), estimated dollar cost since creation of
        :class:`Reporter` object is reported automatically as
        :const:`~syne_tune.constants.ST_WORKER_COST`. This is available for
        SageMaker backend only. Requires ``add_time=True``.
    """

    add_time: bool = True
    add_cost: bool = True

    def __post_init__(self):
        if self.add_time:
            self.start = perf_counter()
            self.iter = 0
            # TODO dollar-cost computation is not available for file-based backends, what would be
            #  needed to add support for those backends will be to add a way to access instance-type
            #  information.
            if self.add_cost:
                # add instance_type and instance count so that cost can be computed easily
                self.instance_type = os.getenv(
                    f"SM_HP_{ST_INSTANCE_TYPE.upper()}", None
                )
                self.instance_count = literal_eval(
                    os.getenv(f"SM_HP_{ST_INSTANCE_COUNT.upper()}", "1")
                )
                if self.instance_type is not None:
                    logger.info(
                        f"detected instance-type/instance-count to {self.instance_type}/{self.instance_count}"
                    )
                    instance_infos = InstanceInfos()
                    if self.instance_type in instance_infos.instances:
                        cost_per_hour = instance_infos(
                            instance_type=self.instance_type
                        ).cost_per_hour
                        self.dollar_cost = cost_per_hour * self.instance_count / 3600

    def __call__(self, **kwargs) -> None:
        """Report metric values from training function back to Syne Tune

        A time stamp :const:`~syne_tune.constants.ST_WORKER_TIMESTAMP` is added.
        See :attr:`add_time`, :attr:`add_cost` comments for other automatically
        added metrics.

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
    print(f"[{ST_SAGEMAKER_METRIC_TAG}]: {_serialize_report_dict(kwargs)}")
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
    regex = r"\[" + ST_SAGEMAKER_METRIC_TAG + r"\]: (\{.*\})"
    for metric_values in re.findall(regex, "\n".join(log_lines)):
        metrics.append(json.loads(metric_values))
    return metrics
