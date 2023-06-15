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
from syne_tune.experiments.benchmark_definitions.yahpo import (
    benchmark_name,
    yahpo_iaml_benchmark,
    yahpo_iaml_methods,
    yahpo_iaml_metrics,
    yahpo_iaml_instances,
    yahpo_iaml_selected_instances,
    yahpo_iaml_benchmark_definitions,
    yahpo_iaml_selected_benchmark_definitions,
    yahpo_iaml_sampled_fidelities,
    yahpo_rbv2_benchmark,
    yahpo_rbv2_methods,
    yahpo_rbv2_metrics,
    yahpo_rbv2_instances,
    yahpo_rbv2_selected_instances,
    yahpo_rbv2_max_wallclock_time,
    yahpo_rbv2_benchmark_definitions,
    yahpo_rbv2_selected_benchmark_definitions,
    yahpo_rbv2_sampled_fidelities,
)


def test_yahpo_iaml():
    assert set(yahpo_iaml_methods) == set(yahpo_iaml_benchmark.keys())
    names_dict_lst = [
        {
            method: {metric: yahpo_iaml_instances for metric, _ in yahpo_iaml_metrics}
            for method in yahpo_iaml_methods
        },
        yahpo_iaml_selected_instances,
    ]
    for i, names in enumerate(names_dict_lst):
        for method in yahpo_iaml_methods:
            if i == 0:
                definitions = yahpo_iaml_benchmark_definitions(
                    method, restrict_fidelities=True
                )
            else:
                definitions = yahpo_iaml_selected_benchmark_definitions(
                    method, restrict_fidelities=True
                )
            for metric, mode in yahpo_iaml_metrics:
                for name in names[method][metric]:
                    errstr = f"{i}, {method}, {metric}, {name}"
                    bm_name = benchmark_name("iaml", method, metric, name)
                    benchmark = definitions.get(bm_name)
                    assert benchmark is not None, errstr
                    assert benchmark.dataset_name == name, errstr
                    assert benchmark.metric == metric, errstr
                    assert benchmark.mode == mode, errstr
                    assert benchmark.blackbox_name == f"yahpo-iaml_{method}", errstr
                    assert benchmark.elapsed_time_attr == "timetrain", errstr
                    assert benchmark.surrogate_kwargs is not None, errstr
                    lst = benchmark.surrogate_kwargs.get("fidelities")
                    assert lst is not None, errstr
                    assert set(lst) == set(yahpo_iaml_sampled_fidelities)


def test_yahpo_rbv2():
    assert set(yahpo_rbv2_methods) == set(yahpo_rbv2_benchmark.keys())
    names_dict_lst = [yahpo_rbv2_instances, yahpo_rbv2_selected_instances]
    for i, names in enumerate(names_dict_lst):
        for method in yahpo_rbv2_methods:
            if i == 0:
                definitions = yahpo_rbv2_benchmark_definitions(
                    method, restrict_fidelities=True
                )
            else:
                definitions = yahpo_rbv2_selected_benchmark_definitions(
                    method, restrict_fidelities=True
                )
            for metric, mode in yahpo_rbv2_metrics:
                for name in names[method][metric]:
                    errstr = f"{method}, {metric}, {name}"
                    bm_name = benchmark_name("rbv2", method, metric, name)
                    benchmark = definitions.get(bm_name)
                    assert benchmark is not None, errstr
                    assert benchmark.dataset_name == name, errstr
                    assert benchmark.metric == metric, errstr
                    assert benchmark.mode == mode, errstr
                    assert benchmark.blackbox_name == f"yahpo-rbv2_{method}", errstr
                    assert benchmark.elapsed_time_attr == "timetrain", errstr
                    assert benchmark.surrogate_kwargs is not None, errstr
                    lst = benchmark.surrogate_kwargs.get("fidelities")
                    assert lst is not None, errstr
                    assert set(lst) == set(yahpo_rbv2_sampled_fidelities)
                    k = f"{method}-{metric}-{name}"
                    assert k in yahpo_rbv2_max_wallclock_time, errstr
                    assert (
                        benchmark.max_wallclock_time == yahpo_rbv2_max_wallclock_time[k]
                    )
