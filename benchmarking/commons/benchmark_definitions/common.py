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


@dataclass
class SurrogateBenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_resource_attr: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    surrogate_kwargs: Optional[dict] = None
    datasets: Optional[List[str]] = None


@dataclass
class RealBenchmarkDefinition:
    script: Path
    config_space: dict
    max_wallclock_time: float
    n_workers: int
    instance_type: str
    metric: str
    mode: str
    max_resource_attr: str
    resource_attr: Optional[str] = None
    framework: Optional[str] = None
    estimator_kwargs: Optional[dict] = None
    max_num_evaluations: Optional[int] = None
