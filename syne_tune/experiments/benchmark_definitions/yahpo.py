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
from typing import Dict, Any
import json
from pathlib import Path

from syne_tune.experiments.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)
from syne_tune.experiments.benchmark_definitions.lcbench import (
    lcbench_selected_datasets,
)


# Note: We do not include scenarios ``iaml_super``, ``rbv2_super`` for now,
# because they need proper usage of conditional configuration spaces,
# which is not supported yet.

# Note: We do not include the ``fcnet`` scenario. FCNet is a tabulated benchmark
# evaluated completely on a fine grid, so does not profit from surrogate
# modelling. Our own implementation works fine and provides for much faster
# simulations.


# -----
# nb301
# -----


def yahpo_nb301_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=12 * 3600,
        n_workers=4,
        elapsed_time_attr="runtime",
        metric="val_accuracy",
        mode="max",
        blackbox_name="yahpo-nb301",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_nb301_instances = ["CIFAR10"]


yahpo_nb301_benchmark_definitions = {
    "yahpo-nb301-" + name: yahpo_nb301_benchmark(name) for name in yahpo_nb301_instances
}


# -------
# lcbench
# -------


# ``openml_task_name_to_id`` maps OpenML task name to task ID
with open(Path(__file__).parent / "openml_task_name_to_id.json", "r") as fp:
    openml_task_name_to_id = json.load(fp)


def yahpo_lcbench_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="yahpo-lcbench",
        dataset_name=openml_task_name_to_id[dataset_name],
        max_resource_attr="max_resources",
    )


yahpo_lcbench_instances = list(openml_task_name_to_id.keys())


yahpo_lcbench_benchmark_definitions = {
    "yahpo-lcbench-" + name: yahpo_lcbench_benchmark(name)
    for name in yahpo_lcbench_instances
}


yahpo_lcbench_selected_benchmark_definitions = {
    "yahpo-lcbench-" + name: yahpo_lcbench_benchmark(name)
    for name in lcbench_selected_datasets
}


# ----
# iaml
# ----


def benchmark_name(scenario, method, metric, name):
    return f"yahpo-{scenario}_{method}_{metric}-{name}"


yahpo_iaml_metrics = (("f1", "max"), ("auc", "max"))


yahpo_iaml_sampled_fidelities = [1, 2, 4, 8, 12, 16, 20]


yahpo_iaml_methods = ["rpart", "glmnet", "ranger", "xgboost"]


def yahpo_iaml_rpart_benchmark(
    dataset_name, metric, mode="max", restrict_fidelities=False
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_iaml_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=240,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_rpart",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_iaml_glmnet_benchmark(
    dataset_name, metric, mode="max", restrict_fidelities=False
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_iaml_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=180,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_glmnet",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_iaml_ranger_benchmark(
    dataset_name, metric, mode="max", restrict_fidelities=False
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_iaml_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=300,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_ranger",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_iaml_xgboost_benchmark(
    dataset_name, metric, mode="max", restrict_fidelities=False
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_iaml_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=300,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_xgboost",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


yahpo_iaml_benchmark = {
    "rpart": yahpo_iaml_rpart_benchmark,
    "glmnet": yahpo_iaml_glmnet_benchmark,
    "ranger": yahpo_iaml_ranger_benchmark,
    "xgboost": yahpo_iaml_xgboost_benchmark,
}


yahpo_iaml_instances = ["40981", "41146", "1489", "1067"]


yahpo_iaml_selected_instances = {
    "rpart": {
        "f1": ["41146", "1489", "1067"],
        "auc": ["41146", "1489", "1067"],
    },
    "glmnet": {
        "f1": ["1489", "1067"],
        "auc": ["1489"],
    },
    "ranger": {
        "f1": ["41146", "1489", "1067"],
        "auc": ["41146", "1489"],
    },
    "xgboost": {
        "f1": ["40981", "1489"],
        "auc": ["40981", "1489"],
    },
}


def _yahpo_iaml_benchmark_definitions(
    instances_dict, method, restrict_fidelities=False
) -> Dict[str, Any]:
    assert (
        method in yahpo_iaml_benchmark
    ), f"method = {method} must be in {list(yahpo_iaml_benchmark.keys())}"
    return {
        benchmark_name("iaml", method, metric, name): yahpo_iaml_benchmark[method](
            name,
            metric,
            mode,
            restrict_fidelities=restrict_fidelities,
        )
        for metric, mode in yahpo_iaml_metrics
        for name in instances_dict[method][metric]
    }


def yahpo_iaml_benchmark_definitions(
    method, restrict_fidelities=False
) -> Dict[str, Any]:
    instances_dict = {
        method: {metric: yahpo_iaml_instances for metric, _ in yahpo_iaml_metrics}
        for method in yahpo_iaml_methods
    }
    return _yahpo_iaml_benchmark_definitions(
        instances_dict, method, restrict_fidelities
    )


def yahpo_iaml_selected_benchmark_definitions(
    method, restrict_fidelities=False
) -> Dict[str, Any]:
    return _yahpo_iaml_benchmark_definitions(
        yahpo_iaml_selected_instances, method, restrict_fidelities
    )


# ----
# rbv2
# ----


yahpo_rbv2_metrics = (("f1", "max"), ("auc", "max"))


yahpo_rbv2_sampled_fidelities = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]


yahpo_rbv2_methods = ["rpart", "glmnet", "ranger", "xgboost", "svm", "aknn"]


def yahpo_rbv2_rpart_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_rpart",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_rbv2_glmnet_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_glmnet",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_rbv2_ranger_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_ranger",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_rbv2_xgboost_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_xgboost",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_rbv2_svm_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=1000,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_svm",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


def yahpo_rbv2_aknn_benchmark(
    dataset_name,
    metric,
    mode="max",
    restrict_fidelities=False,
    max_wallclock_time=900,
):
    surrogate_kwargs = (
        {
            "fidelities": yahpo_rbv2_sampled_fidelities,
        }
        if restrict_fidelities
        else None
    )
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=max_wallclock_time,
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_aknn",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
        surrogate_kwargs=surrogate_kwargs,
    )


yahpo_rbv2_benchmark = {
    "rpart": yahpo_rbv2_rpart_benchmark,
    "glmnet": yahpo_rbv2_glmnet_benchmark,
    "ranger": yahpo_rbv2_ranger_benchmark,
    "xgboost": yahpo_rbv2_xgboost_benchmark,
    "svm": yahpo_rbv2_svm_benchmark,
    "aknn": yahpo_rbv2_aknn_benchmark,
}


# ``yahpo_rbv2_instances[method][metric]`` contains list of instances for
# this method and metric.
#
# These are not all instances available in YAHPO rbv2, some filtering has
# been applied:
# - f1:
#   After sampling 100 configs at random, best f1 > 0.2.
#   Statistics of instances filtered out:
#   rpart:   max = 0.011020094156265259, avg = 0.0008951445925049484
#   glmnet:  max = 0.0017474889755249023, avg = 7.839714089641348e-05
#   ranger:  max = 0.059225380420684814, avg = 0.0039265831001102924
#   xgboost: max = 0.18943724036216736, avg = 0.015040344558656216
#   svm:     max = 0.023529857397079468, avg = 0.0020623435266315937
#   aknn:    max = 0.02313774824142456, avg = 0.0024604059290140867
# - auc:
#   After sampling 1000 configs at random, best auc < 0.999
#   Statistics of instances filtered out:
#   rpart:   -
#   glmnet:  min = 0.9990544319152832, avg = 0.999367892742157
#   ranger:  min = 0.9990753531455994, avg = 0.9997223019599915
#   xgboost: min = 0.9995461702346802, avg = 0.999969482421875
#   svm:     min = 0.999025821685791, avg = 0.9996441006660461
#   aknn:    min = 0.9991077780723572, avg = 0.9997339844703674

with open(Path(__file__).parent / "yahpo_rbv2_instances.json", "r") as fp:
    yahpo_rbv2_instances = json.load(fp)


# ``yahpo_rbv2_selected_instances[method][metric]`` contains list of instances
# selected by visual inspection among those with the largest estimated
# ``max_wallclock_time``.

with open(Path(__file__).parent / "yahpo_rbv2_selected_instances.json", "r") as fp:
    yahpo_rbv2_selected_instances = json.load(fp)


# ``yahpo_rbv2_max_wallclock_time`` maps key of the form
# ``f"{method}-{metric}-{instance}"`` to suggested value for
# ``max_wallclock_time``. Here, ``instance`` loops over
# ``yahpo_rbv2_instances[method][metric]``.
#
# Determined as follows:
# - Ran RS, BO, ASHA with max_wallclock_time = 900
# - Aggregate curves over 5 seeds
# - For each instance: minmax = max minus min over all metric values;
#   eps = 0.01 * minmax
# - For each HPO method: Smallest time where metric < final_val + eps
#   (here, metric is minus auc or minus f1)
# - max_wallclock_time is then the maximum over these 3 values
# - visual inspection of largest (> 700) and smallest (< 100)
#   - remove if time < 50
#   - remove if flat very soon and all the same
#   - adjust time after visual inspection
# - ``yahpo_rbv2_selected_instances`` from largest upon visual
#    inspection
# For all others, the estimated value is used, and they are not
# considered for ``yahpo_rbv2_selected_instances``
#
# This failed with all 0 for xgboost-auc-*, svm-auc-* (all), so these
# have been removed entirely.

with open(Path(__file__).parent / "yahpo_rbv2_max_wallclock_time.json", "r") as fp:
    yahpo_rbv2_max_wallclock_time = json.load(fp)


def _yahpo_rbv2_benchmark_definitions(
    instances_dict, method, restrict_fidelities=False
) -> Dict[str, Any]:
    assert (
        method in yahpo_rbv2_benchmark
    ), f"method = {method} must be in {list(yahpo_iaml_benchmark.keys())}"
    return {
        benchmark_name("rbv2", method, metric, name): yahpo_rbv2_benchmark[method](
            name,
            metric,
            mode,
            restrict_fidelities=restrict_fidelities,
            max_wallclock_time=yahpo_rbv2_max_wallclock_time[
                f"{method}-{metric}-{name}"
            ],
        )
        for metric, mode in yahpo_rbv2_metrics
        for name in instances_dict[method][metric]
    }


def yahpo_rbv2_benchmark_definitions(
    method, restrict_fidelities=False
) -> Dict[str, Any]:
    return _yahpo_rbv2_benchmark_definitions(
        yahpo_rbv2_instances, method, restrict_fidelities
    )


def yahpo_rbv2_selected_benchmark_definitions(
    method, restrict_fidelities=False
) -> Dict[str, Any]:
    return _yahpo_rbv2_benchmark_definitions(
        yahpo_rbv2_selected_instances, method, restrict_fidelities
    )
