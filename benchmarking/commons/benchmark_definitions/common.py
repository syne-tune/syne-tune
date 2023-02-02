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
from typing import Optional, List, Union, Dict, Any


@dataclass
class SurrogateBenchmarkDefinition:
    """Meta-data for tabulated benchmark, served by the blackbox repository.

    :param max_wallclock_time: Default value for stopping criterion
    :param n_workers: Default value for tuner
    :param elapsed_time_attr: Name of metric reported
    :param metric: Name of metric reported
    :param mode: "max" or "min"
    :param blackbox_name: Name of blackbox, see :func:`load_blackbox`
    :param dataset_name: Dataset (or instance) for blackbox
    :param max_num_evaluations (optional): Default value for stopping criterion
    :param surrogate: Default value for surrogate to be used, see
        :func:`make_surrogate`. Otherwise: use no surrogate
    :param surrogate_kwargs: Default value for arguments of surrogate,
        see :func:`make_surrogate`
    :param add_surrogate_kwargs: Arguments passed to :func:`add_surrogate`. Optional.
    :param max_resource_attr: Internal name between backend and scheduler
    :param datasets: Used in transfer tuning
    :param fidelities: If given, this is a strictly increasing subset of
        the fidelity values provided by the surrogate, and only those
        will be reported
    """

    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    surrogate_kwargs: Optional[dict] = None
    add_surrogate_kwargs: Optional[dict] = None
    max_resource_attr: Optional[str] = None
    datasets: Optional[List[str]] = None
    fidelities: Optional[List[int]] = None

    def __post_init__(self):
        if self.max_resource_attr is None:
            self.max_resource_attr = "epochs"


@dataclass
class RealBenchmarkDefinition:
    """Meta-data for real benchmark, given by code.

    :param script: Absolute filename of training script
    :param config_space: Default value for configuration space, must include
        ``max_resource_attr``
    :param max_wallclock_time: Default value for stopping criterion
    :param n_workers: Default value for tuner
    :param instance_type: Default value for instance type
    :param metric: Name of metric reported
    :param mode: "max" or "min"
    :param max_resource_attr: Name of ``config_space`` entry
    :param framework: SageMaker framework to be used for ``script``. Additional
        dependencies in ``requirements.txt`` in ``script.parent``
    :param resource_attr (optional): Name of attribute reported (required for
        multi-fidelity)
    :param estimator_kwargs (optional): Additional arguments to SageMaker
        estimator, e.g. ``framework_version``
    :param max_num_evaluations (optional): Default value for stopping criterion
    """

    script: Path
    config_space: Dict[str, Any]
    max_wallclock_time: float
    n_workers: int
    instance_type: str
    metric: str
    mode: str
    max_resource_attr: str
    framework: str
    resource_attr: Optional[str] = None
    estimator_kwargs: Optional[dict] = None
    max_num_evaluations: Optional[int] = None


BenchmarkDefinition = Union[SurrogateBenchmarkDefinition, RealBenchmarkDefinition]
