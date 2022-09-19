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
import numpy as np
import json
import logging
from ast import literal_eval
from typing import List, Dict
from time import time, perf_counter
from dataclasses import dataclass

from syne_tune.constants import (
    ST_INSTANCE_TYPE,
    ST_INSTANCE_COUNT,
    ST_WORKER_TIME,
    ST_WORKER_COST,
    ST_WORKER_TIMESTAMP,
    ST_WORKER_ITER,
)

# this is required so that metrics are written
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos

logging.basicConfig()
logger = logging.getLogger(__name__)


@dataclass
class Reporter:
    # Whether to add automatically `st_worker_time` information on the metric reported by the worker which measures
    # the number of seconds spent since the creation of the Reporter.
    add_time: bool = True

    # Whether to add automatically `st_worker_cost` information on the metric reported by the worker which measures
    # the estimated dollar-cost (only available and activated by default on Sagemaker backend). This option requires
    # add_time to be activated.
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

    def __call__(self, **kw) -> None:
        """
        Report metrics obtained after evaluating a training function, see `train_height.py` for a full
         example on how to use it and to define a training function.
        A time stamp `st_timestamp` is added, if `add_time` is True then `st_worker_time` is added
        which measures seconds spent in the worker, if `add_cost` is True `st_worker_cost` is added that measures
        dollar cost spent in worker (only available on Sagemaker instances and requires `st_worker_time` to be
        activated).
        :param kwargs: key word arguments of the metrics to report, for instance `report(epoch=1, loss=1.2)` the only
        constrain is that values must be serializable with json and keys should not start with `st_` which is a
        reserved namespace for Syne Tune internals.
        """
        for key in kw.keys():
            assert not key.startswith("st_"), (
                "The metric prefix 'st_' is used by Syne Tune internals, "
                "please use a metric name that does not start with 'st_'."
            )

        kw[ST_WORKER_TIMESTAMP] = time()
        if self.add_time:
            seconds_spent = perf_counter() - self.start
            kw[ST_WORKER_TIME] = seconds_spent
            # second cost will only be there if we were able to properly detect the instance-type and instance-count
            # from the environment
            if hasattr(self, "dollar_cost"):
                kw[ST_WORKER_COST] = seconds_spent * self.dollar_cost
        kw[ST_WORKER_ITER] = self.iter
        self.iter += 1
        _report_logger(**kw)


def _report_logger(**kwargs):
    print(f"[tune-metric]: {_serialize_report_dict(kwargs)}")
    sys.stdout.flush()


def _serialize_report_dict(report_dict: dict) -> str:
    """
    :param report_dict: a dictionary of metrics to be serialized
    :return: serialized string of the reported metrics, an exception is raised if the size is too large or
    if the dictionary values are not JSON-serializable
    """
    try:

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        report_str = json.dumps(report_dict, default=np_encoder)
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
    """
    Retrieves metrics reported with `_report_logger` given log lines.
    :param log_lines:
    :return: list of metrics retrieved from the log lines.
    """
    metrics = []
    regex = r"\[tune-metric\]: (\{.*\})"
    for metric_values in re.findall(regex, "\n".join(log_lines)):
        metrics.append(json.loads(metric_values))
    return metrics
