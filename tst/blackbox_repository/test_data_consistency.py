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
from typing import Optional, List
import pytest
import numpy as np

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    make_surrogate,
)
from syne_tune.blackbox_repository import load_blackbox, add_surrogate
from syne_tune.blackbox_repository.utils import metrics_for_configuration
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.optimizer.schedulers.searchers.searcher import RandomSearcher


@dataclass
class BenchmarkDefinition:
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_resource_attr: str
    surrogate: Optional[str] = None
    surrogate_kwargs: Optional[dict] = None


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


def lcbench_benchmark(dataset_name):
    return BenchmarkDefinition(
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        max_resource_attr="epochs",
    )


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}

# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions[
        "lcbench-" + task.replace("_", "-").replace(".", "")
    ] = lcbench_benchmark(task)


def create_blackbox(benchmark: BenchmarkDefinition):
    # See also :class:`BlackboxRepositoryBackend`
    blackbox = load_blackbox(benchmark.blackbox_name)[benchmark.dataset_name]
    if benchmark.surrogate is not None:
        surrogate = make_surrogate(benchmark.surrogate, benchmark.surrogate_kwargs)
        blackbox = add_surrogate(blackbox=blackbox, surrogate=surrogate)
    return blackbox


def _assert_strictly_increasing(elapsed_times: List[float], error_prefix: str):
    error_msg_parts = []
    for pos, (et1, et2) in enumerate(zip(elapsed_times[:-1], elapsed_times[1:])):
        if et1 >= et2:
            error_msg_parts.append(f"{pos + 1}:{et1} -> {pos + 2}:{et2}")
    if error_msg_parts:
        error_msg = error_prefix + "\n" + ", ".join(error_msg_parts)
    else:
        error_msg = ""
    assert not error_msg, error_msg


def _assert_no_extreme_deviations(elapsed_times: List[float], error_prefix: str):
    pairs = list(zip(elapsed_times[:-1], elapsed_times[1:]))
    epoch_times = [et2 - et1 for et1, et2 in pairs]
    median_val = np.median(epoch_times)
    error_msg_parts = []
    for pos, (val, (et1, et2)) in enumerate(zip(epoch_times, pairs)):
        if val < 0.01 * median_val or val > 100 * median_val:
            error_msg_parts.append(f"{pos + 1}:{et1} -> {pos + 2}:{et2}")
    if error_msg_parts:
        error_msg = (
            error_prefix
            + f"\nmedian_diff = {median_val}: "
            + ", ".join(error_msg_parts)
        )
    else:
        error_msg = ""
    assert not error_msg, error_msg


@pytest.mark.skip("Needs blackbox data files locally or on S3")
@pytest.mark.parametrize("benchmark", benchmark_definitions.values())
def test_elapsed_time_consistency(benchmark):
    num_configs = 20
    random_seed = 382378624

    blackbox = create_blackbox(benchmark)
    resource_attr = next(iter(blackbox.fidelity_space.keys()))
    elapsed_time_attr = benchmark.elapsed_time_attr
    num_fidelities = len(blackbox.fidelity_values)
    if isinstance(blackbox, BlackboxTabular):
        seeds = list(range(blackbox.num_seeds))
    else:
        seeds = [None]
    random_searcher = RandomSearcher(
        config_space=blackbox.configuration_space,
        metric=benchmark.metric,
        random_seed=random_seed,
    )
    configs = [random_searcher.get_config() for _ in range(num_configs)]
    for config in configs:
        for seed in seeds:
            error_prefix = (
                f"blackbox = {benchmark.blackbox_name}, "
                f"dataset = {benchmark.dataset_name}, "
                f"seed = {seed}, config = {config}"
            )
            all_results = metrics_for_configuration(
                blackbox=blackbox,
                config=config,
                resource_attr=resource_attr,
                seed=seed,
            )
            assert len(all_results) == num_fidelities, error_prefix
            elapsed_times = [np.nan] * num_fidelities
            for result in all_results:
                resource = int(result[resource_attr])
                elapsed_time = float(result[elapsed_time_attr])
                assert 1 <= resource <= num_fidelities, (
                    error_prefix + f", result = {result}"
                )
                elapsed_times[resource - 1] = elapsed_time
            assert not any([np.isnan(x) for x in elapsed_times]), (
                error_prefix + f", elapsed_times = {elapsed_times}"
            )
            # elapsed_times must be strictly increasing
            _assert_strictly_increasing(elapsed_times, error_prefix)
            # No extreme deviation from median in per epoch times
            _assert_no_extreme_deviations(elapsed_times, error_prefix)
