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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd


@dataclass
class InstanceInfo:
    name: str
    num_cpu: int
    num_gpu: int
    cost_per_hour: float


class InstanceInfos:
    """
    Utility to get information of an instance type (num cpu/gpu, cost per hour).
    """

    def __init__(self):
        # TODO right now, we use a static file but some services are available to get updated information
        root = Path(__file__).parent
        self.df_instances = pd.read_csv(
            root / "instance-types-cost.csv", delimiter=";"
        ).sort_values(by="price")
        self.instances = list(self.df_instances.instance.unique())

    def __call__(self, instance_type: str) -> InstanceInfo:
        row = self.df_instances.loc[self.df_instances.instance == instance_type]
        return InstanceInfo(
            name=row["instance"].values[0],
            num_cpu=row["vCPU"].values[0],
            num_gpu=row["GPU"].values[0],
            cost_per_hour=row["price"].values[0],
        )


def select_instance_type(
    min_gpu: int = 0,
    max_gpu: int = 16,
    min_cost_per_hour: Optional[float] = None,
    max_cost_per_hour: Optional[float] = None,
) -> List[str]:
    """
    :param min_gpu:
    :param max_gpu:
    :param min_cost_per_hour:
    :param max_cost_per_hour:
    :return: a list of instance type that met the required constrain on minimum/maximum number of GPU and
    minimum/maximum cost per hour.
    """
    res = []
    instance_infos = InstanceInfos()
    for instance in instance_infos.instances:
        instance_info = instance_infos(instance)
        if instance_info.num_gpu < min_gpu or instance_info.num_gpu > max_gpu:
            continue
        if (
            min_cost_per_hour is not None
            and instance_info.cost_per_hour <= min_cost_per_hour
        ):
            continue
        if (
            max_cost_per_hour is not None
            and instance_info.cost_per_hour >= max_cost_per_hour
        ):
            continue
        res.append(instance)
    return res


if __name__ == "__main__":
    info = InstanceInfos()

    for instance in info.instances:
        print(instance, info(instance))
