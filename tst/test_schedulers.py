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
import pytest
import itertools

from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband_impl import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune.config_space import choice
from tst.util_test import run_experiment_with_height


def make_async_scheduler(scheduler, searcher):
    def maker(
        config_space, metric, mode, random_seed, resource_attr, max_resource_attr
    ):
        search_options = {"debug_log": False}
        if searcher == "hypertune":
            search_options["model"] = "gp_independent"
        if searcher == "grid":
            config_space = dict(
                config_space,
                width=choice([1, 2, 3, 4, 5]),
                height=choice([-3, -2, -1, 0, 1, 2, 3]),
            )
        if scheduler == "fifo":
            myscheduler = FIFOScheduler(
                config_space,
                searcher=searcher,
                search_options=search_options,
                mode=mode,
                metric=metric,
                random_seed=random_seed,
            )
        else:
            prefix = "hyperband_"
            assert scheduler.startswith(prefix)
            sch_type = scheduler[len(prefix) :]
            myscheduler = HyperbandScheduler(
                config_space,
                searcher=searcher,
                search_options=search_options,
                max_resource_attr=max_resource_attr,
                type=sch_type,
                resource_attr=resource_attr,
                random_seed=random_seed,
                mode=mode,
                metric=metric,
            )
        return myscheduler

    return maker


def make_sync_scheduler(scheduler_cls, searcher):
    def maker(
        config_space, metric, mode, random_seed, resource_attr, max_resource_attr
    ):
        search_options = {"debug_log": False}
        if searcher == "grid":
            config_space = dict(
                config_space,
                width=choice([1, 2, 3, 4, 5]),
                height=choice([-3, -2, -1, 0, 1, 2, 3]),
            )
        scheduler_kwargs = dict(
            searcher=searcher,
            search_options=search_options,
            mode=mode,
            metric=metric,
            resource_attr=resource_attr,
            max_resource_attr=max_resource_attr,
            random_seed=random_seed,
        )
        return scheduler_cls(config_space, **scheduler_kwargs)

    return maker


_async_parameterizations = list(
    itertools.product(
        ["fifo", "hyperband_stopping", "hyperband_promotion"],
        ["random", "bayesopt", "grid"],
        ["min", "max"],
    )
) + list(
    itertools.product(
        ["hyperband_stopping", "hyperband_promotion"],
        ["hypertune"],
        ["min", "max"],
    )
)


@pytest.mark.parametrize("scheduler, searcher, mode", _async_parameterizations)
def test_async_scheduler_local(scheduler, searcher, mode):
    run_experiment_with_height(
        make_scheduler=make_async_scheduler(scheduler, searcher),
        simulated=False,
        mode=mode,
    )


@pytest.mark.parametrize("scheduler, searcher, mode", _async_parameterizations)
def test_async_scheduler_simulated(scheduler, searcher, mode):
    run_experiment_with_height(
        make_scheduler=make_async_scheduler(scheduler, searcher),
        simulated=True,
        mode=mode,
    )


_sync_parameterizations = [
    [SynchronousGeometricHyperbandScheduler, "random", "min"],
    [SynchronousGeometricHyperbandScheduler, "random", "max"],
    [SynchronousGeometricHyperbandScheduler, "bayesopt", "min"],
    [SynchronousGeometricHyperbandScheduler, "bayesopt", "max"],
    [SynchronousGeometricHyperbandScheduler, "grid", "min"],
    [SynchronousGeometricHyperbandScheduler, "grid", "max"],
    [GeometricDifferentialEvolutionHyperbandScheduler, "random", "min"],
    [GeometricDifferentialEvolutionHyperbandScheduler, "random_encoded", "max"],
    [GeometricDifferentialEvolutionHyperbandScheduler, "random", "min"],
    [GeometricDifferentialEvolutionHyperbandScheduler, "random_encoded", "max"],
]


@pytest.mark.parametrize("scheduler_cls, searcher, mode", _sync_parameterizations)
def test_sync_scheduler_local(scheduler_cls, searcher, mode):
    run_experiment_with_height(
        make_scheduler=make_sync_scheduler(scheduler_cls, searcher),
        simulated=False,
        mode=mode,
    )


@pytest.mark.parametrize("scheduler_cls, searcher, mode", _sync_parameterizations)
def test_sync_scheduler_simulated(scheduler_cls, searcher, mode):
    run_experiment_with_height(
        make_scheduler=make_sync_scheduler(scheduler_cls, searcher),
        simulated=True,
        mode=mode,
    )
