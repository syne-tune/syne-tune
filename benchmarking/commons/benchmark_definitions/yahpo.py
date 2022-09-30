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
from benchmarking.commons.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)
from benchmarking.commons.benchmark_definitions.lcbench import (
    lcbench_selected_datasets,
)


# Note: We do not include scenarios `iaml_super`, `rbv2_super` for now,
# because they need proper usage of conditional configuration spaces,
# which is not supported yet.

# Note: We do not include the `fcnet` scenario. FCNet is a tabulated benchmark
# evaluated completely on a fine grid, so does not profit from surrogate
# modelling. Our own implementation works fine and provides for much faster
# simulations.


# -----
# nb301
# -----


def yahpo_nb301_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
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


openml_task_name_to_id = {
    "KDDCup09_appetency": "3945",
    "covertype": "7593",
    "Amazon_employee_access": "34539",
    "adult": "126025",
    "nomao": "126026",
    "bank-marketing": "126029",
    "shuttle": "146212",
    "Australian": "167104",
    "kr-vs-kp": "167149",
    "mfeat-factors": "167152",
    "credit-g": "167161",
    "vehicle": "167168",
    "kc1": "167181",
    "blood-transfusion-service-center": "167184",
    "cnae-9": "167185",
    "phoneme": "167190",
    "higgs": "167200",
    "connect-4": "167201",
    "helena": "168329",
    "jannis": "168330",
    "volkert": "168331",
    "MiniBooNE": "168335",
    "APSFailure": "168868",
    "christine": "168908",
    "fabert": "168910",
    "airlines": "189354",
    "jasmine": "189862",
    "sylvine": "189865",
    "albert": "189866",
    "dionis": "189873",
    "car": "189905",
    "segment": "189906",
    "Fashion-MNIST": "189908",
    "jungle_chess_2pcs_raw_endgame_complete": "189909",
}


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


def yahpo_iaml_benchmark_definitions(method, restrict_fidelities=False) -> dict:
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
        for name in yahpo_iaml_instances
    }


def yahpo_iaml_selected_benchmark_definitions(
    method, restrict_fidelities=False
) -> dict:
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
        for name in yahpo_iaml_selected_instances[method][metric]
    }


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
yahpo_rbv2_instances = {
    "rpart": {
        "f1": [
            "41138",
            "4134",
            "1220",
            "4154",
            "40978",
            "1111",
            "41150",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "41157",
            "1461",
            "1462",
            "1464",
            "15",
            "41142",
            "40701",
            "40994",
            "31",
            "6332",
            "37",
            "23381",
            "151",
            "41159",
            "23512",
            "1479",
            "1480",
            "41143",
            "1053",
            "1067",
            "1063",
            "41162",
            "1485",
            "1056",
            "334",
            "24",
            "1486",
            "23517",
            "1487",
            "1068",
            "1050",
            "1049",
            "1489",
            "470",
            "1494",
            "41161",
            "312",
            "44",
            "1040",
            "41146",
            "50",
            "40983",
        ],
        "auc": [
            "41138",
            "4135",
            "40981",
            "4134",
            "40927",
            "1220",
            "4154",
            "40923",
            "41163",
            "40996",
            "4538",
            "40978",
            "375",
            "1111",
            "40496",
            "41150",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "1457",
            "469",
            "41157",
            "11",
            "1461",
            "1462",
            "1464",
            "40975",
            "41142",
            "40701",
            "40994",
            "23",
            "40668",
            "29",
            "31",
            "6332",
            "37",
            "4541",
            "23381",
            "151",
            "188",
            "41164",
            "1475",
            "1476",
            "41159",
            "1478",
            "41169",
            "23512",
            "1479",
            "41212",
            "1480",
            "300",
            "41168",
            "41143",
            "1053",
            "41027",
            "1067",
            "1063",
            "41162",
            "6",
            "1485",
            "1056",
            "22",
            "1515",
            "554",
            "334",
            "1486",
            "23517",
            "1493",
            "28",
            "1487",
            "1068",
            "1050",
            "1049",
            "32",
            "1489",
            "470",
            "1494",
            "41161",
            "41165",
            "182",
            "312",
            "1501",
            "40685",
            "42",
            "44",
            "40982",
            "1040",
            "41146",
            "50",
            "41166",
            "307",
            "1497",
            "60",
            "1510",
            "40983",
            "40498",
            "181",
        ],
    },
    "glmnet": {
        "f1": [
            "41138",
            "4134",
            "1220",
            "4154",
            "1111",
            "41150",
            "40900",
            "40536",
            "41156",
            "41157",
            "1461",
            "1464",
            "41142",
            "40701",
            "40994",
            "31",
            "23381",
            "41159",
            "23512",
            "1479",
            "1480",
            "41143",
            "1053",
            "1067",
            "1063",
            "41162",
            "1485",
            "1056",
            "334",
            "23517",
            "1487",
            "1068",
            "1050",
            "1049",
            "1489",
            "470",
            "1494",
            "41161",
            "38",
            "44",
            "40983",
        ],
        "auc": [
            "41138",
            "4135",
            "4134",
            "1220",
            "4154",
            "41163",
            "4538",
            "40978",
            "1111",
            "41150",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "1457",
            "469",
            "41157",
            "40994",
            "23",
            "40668",
            "29",
            "31",
            "6332",
            "4541",
            "23381",
            "151",
            "41164",
            "1475",
            "1476",
            "41159",
            "1478",
            "41169",
            "23512",
            "1479",
            "41212",
            "1480",
            "300",
            "41168",
            "1053",
            "41027",
            "1063",
            "41162",
            "6",
            "1485",
            "1056",
            "14",
            "22",
            "334",
            "1486",
            "23517",
            "41278",
            "1493",
            "1487",
            "1068",
            "1489",
            "470",
            "41161",
            "182",
            "1501",
            "40685",
            "38",
            "40982",
            "41216",
            "41166",
            "40983",
            "40498",
            "181",
            "554",
        ],
    },
    "ranger": {
        "f1": [
            "40981",
            "4134",
            "1220",
            "4154",
            "40978",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "41157",
            "1461",
            "1464",
            "15",
            "41142",
            "40701",
            "40994",
            "29",
            "31",
            "6332",
            "37",
            "23381",
            "151",
            "1479",
            "1480",
            "41143",
            "1053",
            "1067",
            "1063",
            "3",
            "1485",
            "1056",
            "334",
            "1486",
            "1487",
            "1068",
            "1050",
            "1049",
            "1489",
            "470",
            "1494",
            "312",
            "38",
            "44",
            "1040",
            "41146",
            "40983",
            "41138",
            "1111",
            "41159",
            "41162",
            "23517",
            "41161",
            "41150",
            "23512",
        ],
        "auc": [
            "4135",
            "40981",
            "4134",
            "1220",
            "4154",
            "4538",
            "40978",
            "375",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "469",
            "41157",
            "1461",
            "1464",
            "41142",
            "40701",
            "23",
            "40668",
            "29",
            "31",
            "6332",
            "37",
            "23381",
            "151",
            "41164",
            "1475",
            "1478",
            "1479",
            "41212",
            "1480",
            "41143",
            "1053",
            "41027",
            "1067",
            "1063",
            "6",
            "1485",
            "1056",
            "14",
            "22",
            "1515",
            "334",
            "1486",
            "41278",
            "1487",
            "1068",
            "1050",
            "1049",
            "470",
            "1494",
            "182",
            "312",
            "44",
            "41146",
            "40499",
            "54",
            "41216",
            "307",
            "60",
            "40498",
            "181",
            "41163",
            "300",
            "23517",
            "41166",
            "23512",
            "41168",
            "1493",
        ],
    },
    "xgboost": {
        "f1": [
            "41143",
            "470",
            "1487",
            "1461",
            "31",
            "1067",
            "1590",
            "40983",
            "41163",
            "1220",
            "41159",
            "1457",
            "1480",
            "6332",
            "1479",
            "40536",
            "41138",
            "29",
            "1462",
            "1494",
            "40701",
            "44",
            "334",
            "41142",
            "38",
            "1050",
            "23381",
            "41157",
            "15",
            "4134",
            "40981",
            "41156",
            "3",
            "1049",
            "1063",
            "23512",
            "1068",
            "41161",
            "1489",
            "24",
            "23517",
            "1053",
            "312",
            "1510",
            "41162",
            "1464",
            "41146",
            "60",
            "41150",
            "37",
            "1485",
            "4534",
            "151",
            "40978",
            "40994",
            "50",
            "1486",
        ],
        "auc": [],
    },
    "svm": {
        "f1": [
            "40981",
            "40978",
            "41138",
            "41142",
            "29",
            "37",
            "41143",
            "24",
            "41146",
            "1485",
            "1486",
        ],
        "auc": [],
    },
    "aknn": {
        "f1": [
            "41138",
            "40981",
            "4134",
            "1220",
            "4154",
            "40978",
            "1111",
            "41150",
            "4534",
            "40900",
            "40536",
            "41156",
            "1590",
            "41157",
            "1461",
            "1464",
            "15",
            "41142",
            "40701",
            "40994",
            "29",
            "31",
            "6332",
            "37",
            "23381",
            "151",
            "41159",
            "23512",
            "1480",
            "41143",
            "1053",
            "1067",
            "1063",
            "41162",
            "3",
            "1485",
            "1056",
            "334",
            "1486",
            "23517",
            "1487",
            "1068",
            "1050",
            "1049",
            "1489",
            "470",
            "1494",
            "41161",
            "312",
            "38",
            "44",
            "1040",
            "41146",
            "40983",
        ],
        "auc": [
            "41138",
            "40981",
            "4134",
            "40927",
            "1220",
            "4154",
            "41163",
            "40996",
            "4538",
            "40978",
            "1111",
            "41150",
            "4534",
            "40536",
            "41156",
            "1590",
            "1457",
            "469",
            "41157",
            "11",
            "1461",
            "1464",
            "40975",
            "41142",
            "40701",
            "40994",
            "23",
            "40668",
            "29",
            "31",
            "6332",
            "37",
            "4541",
            "40670",
            "23381",
            "151",
            "188",
            "41164",
            "1475",
            "41159",
            "1478",
            "41169",
            "23512",
            "41212",
            "1480",
            "300",
            "41168",
            "41143",
            "1053",
            "41027",
            "1067",
            "1063",
            "41162",
            "3",
            "1056",
            "18",
            "554",
            "1486",
            "23517",
            "41278",
            "1493",
            "1487",
            "1068",
            "1050",
            "1049",
            "1489",
            "470",
            "1494",
            "41161",
            "41165",
            "40984",
            "1501",
            "38",
            "44",
            "46",
            "40982",
            "1040",
            "41146",
            "54",
            "41216",
            "41166",
            "1497",
            "60",
            "40498",
            "181",
            "40923",
        ],
    },
}


yahpo_rbv2_selected_instances = {
    "rpart": {
        "f1": ["1111", "1464", "41157", "41161"],
        "auc": ["1111", "1479", "4541", "40927", "41161"],
    },
    "glmnet": {
        "f1": ["31", "1053", "1111", "41142", "41150", "41157", "41159", "41161"],
        "auc": [
            "1111",
            "1457",
            "4135",
            "4541",
            "23381",
            "23517",
            "41157",
            "41159",
            "41161",
            "41166",
            "41169",
            "41216",
            "41278",
        ],
    },
    "ranger": {
        "f1": [
            "151",
            "1050",
            "1053",
            "1111",
            "1220",
            "1479",
            "1480",
            "1487",
            "23381",
            "40536",
            "41142",
            "41150",
            "41157",
            "41159",
        ],
        "auc": [
            "1220",
            "1461",
            "23517",
            "41166",
            "41168",
            "41142",
            "151",
            "41164",
            "1479",
        ],
    },
    "xgboost": {
        "f1": [
            "29",
            "50",
            "1479",
            "1485",
            "6332",
            "23517",
            "40978",
            "40981",
            "40994",
            "41157",
        ],
        "auc": [],
    },
    "svm": {
        "f1": ["29", "37", "1485", "40978", "41138", "41142", "41143"],
        "auc": [],
    },
    "aknn": {
        "f1": [
            "31",
            "38",
            "1049",
            "1050",
            "1053",
            "1067",
            "1068",
            "1111",
            "1220",
            "1461",
            "1480",
            "1486",
            "1487",
            "1489",
            "1494",
            "4134",
            "4534",
            "40701",
            "41138",
            "41157",
            "41159",
            "41161",
        ],
        "auc": [
            "18",
            "31",
            "181",
            "469",
            "1111",
            "1220",
            "1461",
            "1497",
            "1590",
            "4154",
            "4534",
            "4541",
            "23517",
            "40536",
            "40668",
            "40701",
            "40927",
            "41142",
            "41159",
            "41161",
            "41165",
            "41168",
            "41216",
        ],
    },
}


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
# - `yahpo_rbv2_selected_instances` from largest upon visual
#    inspection
# For all others, the estimated value is used, and they are not
# considered for `yahpo_rbv2_selected_instances`
#
# This failed with all 0 for xgboost-auc-*, svm-auc-* (all), so these
# have been removed entirely.
yahpo_rbv2_max_wallclock_time = {
    "rpart-f1-41138": 120,  # {'RS': 105, 'BO': 169, 'ASHA': 86}
    "rpart-f1-40981": 85,  # {'RS': 51, 'BO': 12, 'ASHA': 85}
    "rpart-f1-4134": 770,  # {'RS': 770, 'BO': 243, 'ASHA': 424}
    "rpart-f1-1220": 476,  # {'RS': 476, 'BO': 130, 'ASHA': 476}
    "rpart-f1-4154": 458,  # {'RS': 316, 'BO': 458, 'ASHA': 316}
    "rpart-f1-40978": 323,  # {'RS': 132, 'BO': 37, 'ASHA': 323}
    "rpart-f1-1111": 900,  # {'RS': 834, 'BO': 51, 'ASHA': 788}
    "rpart-f1-41150": 747,  # {'RS': 390, 'BO': 69, 'ASHA': 747}
    "rpart-f1-4534": 702,  # {'RS': 702, 'BO': 116, 'ASHA': 702}
    "rpart-f1-40900": 636,  # {'RS': 571, 'BO': 133, 'ASHA': 636}
    "rpart-f1-40536": 557,  # {'RS': 557, 'BO': 335, 'ASHA': 557}
    "rpart-f1-41156": 702,  # {'RS': 702, 'BO': 196, 'ASHA': 201}
    "rpart-f1-1590": 637,  # {'RS': 637, 'BO': 324, 'ASHA': 637}
    "rpart-f1-41157": 900,  # {'RS': 805, 'BO': 244, 'ASHA': 642}
    "rpart-f1-1461": 658,  # {'RS': 658, 'BO': 76, 'ASHA': 492}
    "rpart-f1-1462": 256,  # {'RS': 256, 'BO': 29, 'ASHA': 256}
    "rpart-f1-1464": 900,  # {'RS': 896, 'BO': 893, 'ASHA': 896}
    "rpart-f1-15": 256,  # {'RS': 256, 'BO': 100, 'ASHA': 256}
    "rpart-f1-41142": 781,  # {'RS': 781, 'BO': 236, 'ASHA': 419}
    "rpart-f1-40701": 456,  # {'RS': 321, 'BO': 53, 'ASHA': 456}
    "rpart-f1-40994": 456,  # {'RS': 456, 'BO': 59, 'ASHA': 456}
    "rpart-f1-31": 900,  # {'RS': 851, 'BO': 438, 'ASHA': 852}
    "rpart-f1-6332": 120,  # {'RS': 196, 'BO': 36, 'ASHA': 96}
    "rpart-f1-37": 661,  # {'RS': 591, 'BO': 101, 'ASHA': 661}
    "rpart-f1-23381": 776,  # {'RS': 776, 'BO': 419, 'ASHA': 731}
    "rpart-f1-151": 704,  # {'RS': 704, 'BO': 79, 'ASHA': 704}
    "rpart-f1-41159": 759,  # {'RS': 443, 'BO': 759, 'ASHA': 373}
    "rpart-f1-23512": 900,  # {'RS': 836, 'BO': 375, 'ASHA': 655}
    "rpart-f1-1479": 246,  # {'RS': 246, 'BO': 100, 'ASHA': 246}
    "rpart-f1-1480": 900,  # {'RS': 836, 'BO': 672, 'ASHA': 837}
    "rpart-f1-41143": 207,  # {'RS': 207, 'BO': 122, 'ASHA': 207}
    "rpart-f1-1053": 476,  # {'RS': 476, 'BO': 283, 'ASHA': 476}
    "rpart-f1-1067": 900,  # {'RS': 816, 'BO': 637, 'ASHA': 341}
    "rpart-f1-1063": 900,  # {'RS': 851, 'BO': 311, 'ASHA': 851}
    "rpart-f1-41162": 723,  # {'RS': 723, 'BO': 596, 'ASHA': 682}
    "rpart-f1-1485": 465,  # {'RS': 465, 'BO': 28, 'ASHA': 317}
    "rpart-f1-1056": 316,  # {'RS': 316, 'BO': 224, 'ASHA': 316}
    "rpart-f1-334": 900,  # {'RS': 892, 'BO': 341, 'ASHA': 892}
    "rpart-f1-24": 746,  # {'RS': 746, 'BO': 89, 'ASHA': 746}
    "rpart-f1-1486": 573,  # {'RS': 558, 'BO': 104, 'ASHA': 573}
    "rpart-f1-23517": 407,  # {'RS': 407, 'BO': 61, 'ASHA': 403}
    "rpart-f1-1487": 611,  # {'RS': 566, 'BO': 611, 'ASHA': 566}
    "rpart-f1-1068": 565,  # {'RS': 316, 'BO': 565, 'ASHA': 316}
    "rpart-f1-1050": 686,  # {'RS': 686, 'BO': 218, 'ASHA': 686}
    "rpart-f1-1049": 900,  # {'RS': 816, 'BO': 416, 'ASHA': 211}
    "rpart-f1-1489": 900,  # {'RS': 566, 'BO': 115, 'ASHA': 807}
    "rpart-f1-470": 778,  # {'RS': 686, 'BO': 778, 'ASHA': 686}
    "rpart-f1-1494": 900,  # {'RS': 456, 'BO': 337, 'ASHA': 852}
    "rpart-f1-41161": 900,  # {'RS': 485, 'BO': 753, 'ASHA': 881}
    "rpart-f1-312": 571,  # {'RS': 571, 'BO': 203, 'ASHA': 252}
    "rpart-f1-44": 552,  # {'RS': 552, 'BO': 86, 'ASHA': 552}
    "rpart-f1-1040": 256,  # {'RS': 256, 'BO': 85, 'ASHA': 256}
    "rpart-f1-41146": 701,  # {'RS': 701, 'BO': 302, 'ASHA': 216}
    "rpart-f1-50": 456,  # {'RS': 456, 'BO': 24, 'ASHA': 456}
    "rpart-f1-40983": 321,  # {'RS': 236, 'BO': 109, 'ASHA': 321}
    "rpart-auc-41138": 784,  # {'RS': 66, 'BO': 155, 'ASHA': 784}
    "rpart-auc-4135": 642,  # {'RS': 101, 'BO': 642, 'ASHA': 551}
    "rpart-auc-4134": 563,  # {'RS': 563, 'BO': 468, 'ASHA': 537}
    "rpart-auc-40927": 900,  # {'RS': 851, 'BO': 879, 'ASHA': 622}
    "rpart-auc-1220": 542,  # {'RS': 532, 'BO': 542, 'ASHA': 532}
    "rpart-auc-4154": 531,  # {'RS': 531, 'BO': 150, 'ASHA': 531}
    "rpart-auc-40923": 454,  # {'RS': 400, 'BO': 454, 'ASHA': 289}
    "rpart-auc-41163": 358,  # {'RS': 358, 'BO': 172, 'ASHA': 208}
    "rpart-auc-40996": 195,  # {'RS': 66, 'BO': 195, 'ASHA': 55}
    "rpart-auc-4538": 636,  # {'RS': 636, 'BO': 210, 'ASHA': 636}
    "rpart-auc-40978": 310,  # {'RS': 145, 'BO': 37, 'ASHA': 310}
    "rpart-auc-375": 636,  # {'RS': 417, 'BO': 72, 'ASHA': 636}
    "rpart-auc-1111": 900,  # {'RS': 680, 'BO': 595, 'ASHA': 827}
    "rpart-auc-40496": 326,  # {'RS': 326, 'BO': 60, 'ASHA': 326}
    "rpart-auc-41150": 634,  # {'RS': 491, 'BO': 59, 'ASHA': 634}
    "rpart-auc-4534": 653,  # {'RS': 456, 'BO': 653, 'ASHA': 456}
    "rpart-auc-40900": 456,  # {'RS': 456, 'BO': 30, 'ASHA': 456}
    "rpart-auc-40536": 457,  # {'RS': 457, 'BO': 19, 'ASHA': 457}
    "rpart-auc-41156": 636,  # {'RS': 636, 'BO': 48, 'ASHA': 636}
    "rpart-auc-1590": 657,  # {'RS': 657, 'BO': 36, 'ASHA': 657}
    "rpart-auc-1457": 645,  # {'RS': 391, 'BO': 181, 'ASHA': 645}
    "rpart-auc-469": 637,  # {'RS': 636, 'BO': 315, 'ASHA': 637}
    "rpart-auc-41157": 765,  # {'RS': 765, 'BO': 316, 'ASHA': 616}
    "rpart-auc-11": 456,  # {'RS': 456, 'BO': 52, 'ASHA': 456}
    "rpart-auc-1461": 533,  # {'RS': 533, 'BO': 37, 'ASHA': 533}
    "rpart-auc-1462": 211,  # {'RS': 211, 'BO': 18, 'ASHA': 211}
    "rpart-auc-1464": 900,  # {'RS': 897, 'BO': 41, 'ASHA': 897}
    "rpart-auc-40975": 456,  # {'RS': 456, 'BO': 36, 'ASHA': 456}
    "rpart-auc-41142": 636,  # {'RS': 636, 'BO': 341, 'ASHA': 378}
    "rpart-auc-40701": 456,  # {'RS': 456, 'BO': 30, 'ASHA': 456}
    "rpart-auc-40994": 456,  # {'RS': 456, 'BO': 64, 'ASHA': 456}
    "rpart-auc-23": 586,  # {'RS': 586, 'BO': 83, 'ASHA': 586}
    "rpart-auc-40668": 900,  # {'RS': 897, 'BO': 789, 'ASHA': 897}
    "rpart-auc-29": 441,  # {'RS': 441, 'BO': 29, 'ASHA': 441}
    "rpart-auc-31": 532,  # {'RS': 531, 'BO': 354, 'ASHA': 532}
    "rpart-auc-6332": 531,  # {'RS': 531, 'BO': 35, 'ASHA': 531}
    "rpart-auc-37": 256,  # {'RS': 256, 'BO': 249, 'ASHA': 256}
    "rpart-auc-4541": 900,  # {'RS': 798, 'BO': 799, 'ASHA': 798}
    "rpart-auc-23381": 657,  # {'RS': 256, 'BO': 657, 'ASHA': 531}
    "rpart-auc-151": 704,  # {'RS': 704, 'BO': 43, 'ASHA': 704}
    "rpart-auc-188": 456,  # {'RS': 456, 'BO': 41, 'ASHA': 456}
    "rpart-auc-41164": 649,  # {'RS': 649, 'BO': 453, 'ASHA': 645}
    "rpart-auc-1475": 713,  # {'RS': 636, 'BO': 713, 'ASHA': 636}
    "rpart-auc-1476": 446,  # {'RS': 446, 'BO': 84, 'ASHA': 446}
    "rpart-auc-41159": 740,  # {'RS': 740, 'BO': 550, 'ASHA': 402}
    "rpart-auc-1478": 637,  # {'RS': 637, 'BO': 209, 'ASHA': 637}
    "rpart-auc-41169": 759,  # {'RS': 572, 'BO': 159, 'ASHA': 759}
    "rpart-auc-23512": 660,  # {'RS': 494, 'BO': 139, 'ASHA': 774}
    "rpart-auc-1479": 841,  # {'RS': 431, 'BO': 580, 'ASHA': 841}
    "rpart-auc-41212": 600,  # {'RS': 842, 'BO': 29, 'ASHA': 842}
    "rpart-auc-1480": 531,  # {'RS': 531, 'BO': 29, 'ASHA': 531}
    "rpart-auc-300": 149,  # {'RS': 75, 'BO': 69, 'ASHA': 149}
    "rpart-auc-41168": 770,  # {'RS': 770, 'BO': 43, 'ASHA': 683}
    "rpart-auc-41143": 703,  # {'RS': 332, 'BO': 146, 'ASHA': 703}
    "rpart-auc-1053": 532,  # {'RS': 532, 'BO': 236, 'ASHA': 532}
    "rpart-auc-41027": 456,  # {'RS': 456, 'BO': 36, 'ASHA': 456}
    "rpart-auc-1067": 531,  # {'RS': 531, 'BO': 29, 'ASHA': 531}
    "rpart-auc-1063": 456,  # {'RS': 456, 'BO': 55, 'ASHA': 456}
    "rpart-auc-41162": 569,  # {'RS': 569, 'BO': 69, 'ASHA': 542}
    "rpart-auc-6": 116,  # {'RS': 116, 'BO': 30, 'ASHA': 116}
    "rpart-auc-1485": 641,  # {'RS': 641, 'BO': 107, 'ASHA': 641}
    "rpart-auc-1056": 531,  # {'RS': 531, 'BO': 150, 'ASHA': 531}
    "rpart-auc-22": 326,  # {'RS': 326, 'BO': 35, 'ASHA': 326}
    "rpart-auc-1515": 456,  # {'RS': 456, 'BO': 55, 'ASHA': 456}
    "rpart-auc-554": 166,  # {'RS': 166, 'BO': 163, 'ASHA': 104}
    "rpart-auc-334": 531,  # {'RS': 531, 'BO': 138, 'ASHA': 531}
    "rpart-auc-1486": 900,  # {'RS': 463, 'BO': 62, 'ASHA': 898}
    "rpart-auc-23517": 172,  # {'RS': 172, 'BO': 66, 'ASHA': 170}
    "rpart-auc-1493": 332,  # {'RS': 332, 'BO': 54, 'ASHA': 305}
    "rpart-auc-28": 417,  # {'RS': 417, 'BO': 54, 'ASHA': 302}
    "rpart-auc-1487": 900,  # {'RS': 841, 'BO': 151, 'ASHA': 842}
    "rpart-auc-1068": 531,  # {'RS': 531, 'BO': 64, 'ASHA': 531}
    "rpart-auc-1050": 900,  # {'RS': 531, 'BO': 324, 'ASHA': 897}
    "rpart-auc-1049": 360,  # {'RS': 842, 'BO': 29, 'ASHA': 842}
    "rpart-auc-32": 458,  # {'RS': 456, 'BO': 79, 'ASHA': 458}
    "rpart-auc-1489": 456,  # {'RS': 456, 'BO': 53, 'ASHA': 456}
    "rpart-auc-470": 536,  # {'RS': 96, 'BO': 205, 'ASHA': 536}
    "rpart-auc-1494": 456,  # {'RS': 456, 'BO': 60, 'ASHA': 456}
    "rpart-auc-41161": 900,  # {'RS': 795, 'BO': 564, 'ASHA': 423}
    "rpart-auc-41165": 514,  # {'RS': 514, 'BO': 495, 'ASHA': 261}
    "rpart-auc-182": 636,  # {'RS': 551, 'BO': 221, 'ASHA': 636}
    "rpart-auc-312": 456,  # {'RS': 456, 'BO': 36, 'ASHA': 456}
    "rpart-auc-1501": 456,  # {'RS': 456, 'BO': 73, 'ASHA': 456}
    "rpart-auc-40685": 456,  # {'RS': 456, 'BO': 67, 'ASHA': 456}
    "rpart-auc-42": 456,  # {'RS': 456, 'BO': 305, 'ASHA': 456}
    "rpart-auc-44": 457,  # {'RS': 457, 'BO': 36, 'ASHA': 457}
    "rpart-auc-40982": 456,  # {'RS': 456, 'BO': 53, 'ASHA': 456}
    "rpart-auc-1040": 256,  # {'RS': 256, 'BO': 35, 'ASHA': 256}
    "rpart-auc-41146": 456,  # {'RS': 456, 'BO': 82, 'ASHA': 456}
    "rpart-auc-50": 456,  # {'RS': 456, 'BO': 25, 'ASHA': 456}
    "rpart-auc-41166": 900,  # {'RS': 835, 'BO': 133, 'ASHA': 544}
    "rpart-auc-307": 456,  # {'RS': 456, 'BO': 41, 'ASHA': 456}
    "rpart-auc-1497": 256,  # {'RS': 256, 'BO': 72, 'ASHA': 256}
    "rpart-auc-60": 571,  # {'RS': 571, 'BO': 42, 'ASHA': 551}
    "rpart-auc-1510": 256,  # {'RS': 256, 'BO': 23, 'ASHA': 256}
    "rpart-auc-40983": 456,  # {'RS': 456, 'BO': 41, 'ASHA': 456}
    "rpart-auc-40498": 900,  # {'RS': 656, 'BO': 427, 'ASHA': 897}
    "rpart-auc-181": 456,  # {'RS': 456, 'BO': 63, 'ASHA': 456}
    "glmnet-f1-41138": 540,  # {'RS': 713, 'BO': 648, 'ASHA': 348}
    "glmnet-f1-4134": 167,  # {'RS': 70, 'BO': 167, 'ASHA': 82}
    "glmnet-f1-1220": 538,  # {'RS': 538, 'BO': 60, 'ASHA': 534}
    "glmnet-f1-4154": 298,  # {'RS': 298, 'BO': 39, 'ASHA': 178}
    "glmnet-f1-1111": 900,  # {'RS': 627, 'BO': 892, 'ASHA': 627}
    "glmnet-f1-41150": 900,  # {'RS': 562, 'BO': 789, 'ASHA': 845}
    "glmnet-f1-40900": 136,  # {'RS': 136, 'BO': 23, 'ASHA': 136}
    "glmnet-f1-40536": 521,  # {'RS': 521, 'BO': 69, 'ASHA': 203}
    "glmnet-f1-41156": 480,  # {'RS': 83, 'BO': 641, 'ASHA': 112}
    "glmnet-f1-41157": 900,  # {'RS': 621, 'BO': 314, 'ASHA': 315}
    "glmnet-f1-1461": 326,  # {'RS': 273, 'BO': 326, 'ASHA': 182}
    "glmnet-f1-1464": 411,  # {'RS': 186, 'BO': 42, 'ASHA': 411}
    "glmnet-f1-41142": 900,  # {'RS': 691, 'BO': 551, 'ASHA': 246}
    "glmnet-f1-40701": 417,  # {'RS': 417, 'BO': 81, 'ASHA': 417}
    "glmnet-f1-40994": 103,  # {'RS': 91, 'BO': 103, 'ASHA': 86}
    "glmnet-f1-31": 480,  # {'RS': 171, 'BO': 35, 'ASHA': 783}
    "glmnet-f1-37": 60,  # {'RS': 33, 'BO': 30, 'ASHA': 33}
    "glmnet-f1-23381": 480,  # {'RS': 256, 'BO': 826, 'ASHA': 256}
    "glmnet-f1-41159": 900,  # {'RS': 344, 'BO': 868, 'ASHA': 602}
    "glmnet-f1-23512": 35,  # {'RS': 35, 'BO': 31, 'ASHA': 22}
    "glmnet-f1-1479": 453,  # {'RS': 453, 'BO': 35, 'ASHA': 2}
    "glmnet-f1-1480": 784,  # {'RS': 146, 'BO': 24, 'ASHA': 784}
    "glmnet-f1-41143": 96,  # {'RS': 96, 'BO': 79, 'ASHA': 57}
    "glmnet-f1-1053": 900,  # {'RS': 783, 'BO': 190, 'ASHA': 783}
    "glmnet-f1-1067": 453,  # {'RS': 453, 'BO': 179, 'ASHA': 236}
    "glmnet-f1-1063": 392,  # {'RS': 37, 'BO': 23, 'ASHA': 392}
    "glmnet-f1-41162": 480,  # {'RS': 260, 'BO': 285, 'ASHA': 658}
    "glmnet-f1-1485": 85,  # {'RS': 85, 'BO': 58, 'ASHA': 11}
    "glmnet-f1-1056": 327,  # {'RS': 197, 'BO': 48, 'ASHA': 327}
    "glmnet-f1-334": 326,  # {'RS': 326, 'BO': 186, 'ASHA': 221}
    "glmnet-f1-23517": 77,  # {'RS': 77, 'BO': 12, 'ASHA': 58}
    "glmnet-f1-1487": 480,  # {'RS': 772, 'BO': 41, 'ASHA': 372}
    "glmnet-f1-1068": 464,  # {'RS': 436, 'BO': 464, 'ASHA': 436}
    "glmnet-f1-1050": 206,  # {'RS': 206, 'BO': 34, 'ASHA': 86}
    "glmnet-f1-1049": 348,  # {'RS': 186, 'BO': 29, 'ASHA': 348}
    "glmnet-f1-1489": 900,  # {'RS': 784, 'BO': 352, 'ASHA': 784}
    "glmnet-f1-470": 162,  # {'RS': 162, 'BO': 46, 'ASHA': 111}
    "glmnet-f1-1494": 72,  # {'RS': 72, 'BO': 30, 'ASHA': 72}
    "glmnet-f1-41161": 900,  # {'RS': 790, 'BO': 789, 'ASHA': 672}
    "glmnet-f1-38": 317,  # {'RS': 317, 'BO': 18, 'ASHA': 317}
    "glmnet-f1-44": 72,  # {'RS': 72, 'BO': 20, 'ASHA': 72}
    "glmnet-f1-40983": 900,  # {'RS': 822, 'BO': 652, 'ASHA': 822}
    "glmnet-auc-41138": 900,  # {'RS': 686, 'BO': 608, 'ASHA': 382}
    "glmnet-auc-4135": 900,  # {'RS': 501, 'BO': 610, 'ASHA': 850}
    "glmnet-auc-4134": 900,  # {'RS': 107, 'BO': 750, 'ASHA': 256}
    "glmnet-auc-1220": 532,  # {'RS': 403, 'BO': 532, 'ASHA': 404}
    "glmnet-auc-4154": 298,  # {'RS': 298, 'BO': 69, 'ASHA': 183}
    "glmnet-auc-41163": 722,  # {'RS': 682, 'BO': 722, 'ASHA': 453}
    "glmnet-auc-4538": 171,  # {'RS': 171, 'BO': 81, 'ASHA': 142}
    "glmnet-auc-1111": 900,  # {'RS': 657, 'BO': 817, 'ASHA': 850}
    "glmnet-auc-41150": 298,  # {'RS': 248, 'BO': 298, 'ASHA': 289}
    "glmnet-auc-40536": 140,  # {'RS': 140, 'BO': 34, 'ASHA': 129}
    "glmnet-auc-1590": 80,  # {'RS': 28, 'BO': 51, 'ASHA': 19}
    "glmnet-auc-1457": 900,  # {'RS': 776, 'BO': 617, 'ASHA': 685}
    "glmnet-auc-469": 582,  # {'RS': 111, 'BO': 582, 'ASHA': 209}
    "glmnet-auc-41157": 600,  # {'RS': 621, 'BO': 142, 'ASHA': 311}
    "glmnet-auc-41142": 343,  # {'RS': 343, 'BO': 258, 'ASHA': 190}
    "glmnet-auc-40701": 56,  # {'RS': 56, 'BO': 35, 'ASHA': 56}
    "glmnet-auc-23": 166,  # {'RS': 166, 'BO': 18, 'ASHA': 166}
    "glmnet-auc-40668": 375,  # {'RS': 375, 'BO': 279, 'ASHA': 293}
    "glmnet-auc-31": 279,  # {'RS': 86, 'BO': 279, 'ASHA': 103}
    "glmnet-auc-6332": 179,  # {'RS': 179, 'BO': 59, 'ASHA': 158}
    "glmnet-auc-4541": 695,  # {'RS': 695, 'BO': 471, 'ASHA': 602}
    "glmnet-auc-23381": 720,  # {'RS': 271, 'BO': 597, 'ASHA': 186}
    "glmnet-auc-151": 80,  # {'RS': 55, 'BO': 37, 'ASHA': 50}
    "glmnet-auc-41164": 557,  # {'RS': 557, 'BO': 458, 'ASHA': 444}
    "glmnet-auc-1475": 272,  # {'RS': 272, 'BO': 65, 'ASHA': 272}
    "glmnet-auc-41159": 900,  # {'RS': 511, 'BO': 751, 'ASHA': 741}
    "glmnet-auc-1478": 51,  # {'RS': 46, 'BO': 37, 'ASHA': 51}
    "glmnet-auc-41169": 900,  # {'RS': 843, 'BO': 856, 'ASHA': 507}
    "glmnet-auc-23512": 228,  # {'RS': 228, 'BO': 90, 'ASHA': 118}
    "glmnet-auc-1479": 287,  # {'RS': 287, 'BO': 18, 'ASHA': 122}
    "glmnet-auc-1480": 149,  # {'RS': 103, 'BO': 149, 'ASHA': 103}
    "glmnet-auc-300": 595,  # {'RS': 595, 'BO': 170, 'ASHA': 272}
    "glmnet-auc-41168": 720,  # {'RS': 796, 'BO': 395, 'ASHA': 603}
    "glmnet-auc-1053": 117,  # {'RS': 117, 'BO': 37, 'ASHA': 117}
    "glmnet-auc-41027": 110,  # {'RS': 110, 'BO': 43, 'ASHA': 78}
    "glmnet-auc-41162": 720,  # {'RS': 642, 'BO': 613, 'ASHA': 361}
    "glmnet-auc-6": 132,  # {'RS': 132, 'BO': 92, 'ASHA': 84}
    "glmnet-auc-1485": 900,  # {'RS': 893, 'BO': 421, 'ASHA': 360}
    "glmnet-auc-1056": 582,  # {'RS': 582, 'BO': 18, 'ASHA': 431}
    "glmnet-auc-334": 557,  # {'RS': 556, 'BO': 40, 'ASHA': 557}
    "glmnet-auc-23517": 600,  # {'RS': 679, 'BO': 486, 'ASHA': 495}
    "glmnet-auc-41278": 720,  # {'RS': 682, 'BO': 307, 'ASHA': 303}
    "glmnet-auc-1493": 114,  # {'RS': 107, 'BO': 114, 'ASHA': 102}
    "glmnet-auc-1487": 282,  # {'RS': 282, 'BO': 23, 'ASHA': 278}
    "glmnet-auc-1068": 277,  # {'RS': 277, 'BO': 104, 'ASHA': 277}
    "glmnet-auc-470": 401,  # {'RS': 31, 'BO': 29, 'ASHA': 401}
    "glmnet-auc-41161": 720,  # {'RS': 756, 'BO': 750, 'ASHA': 428}
    "glmnet-auc-40685": 261,  # {'RS': 238, 'BO': 261, 'ASHA': 211}
    "glmnet-auc-41216": 720,  # {'RS': 390, 'BO': 257, 'ASHA': 694}
    "glmnet-auc-41166": 900,  # {'RS': 794, 'BO': 865, 'ASHA': 826}
    "glmnet-auc-1497": 47,  # {'RS': 35, 'BO': 47, 'ASHA': 35}
    "glmnet-auc-40983": 277,  # {'RS': 277, 'BO': 24, 'ASHA': 277}
    "glmnet-auc-40498": 201,  # {'RS': 182, 'BO': 132, 'ASHA': 201}
    "glmnet-auc-181": 172,  # {'RS': 172, 'BO': 105, 'ASHA': 125}
    "glmnet-auc-554": 425,  # {'RS': 375, 'BO': 288, 'ASHA': 425}
    "ranger-f1-40981": 443,  # {'RS': 353, 'BO': 443, 'ASHA': 187}
    "ranger-f1-4134": 562,  # {'RS': 562, 'BO': 327, 'ASHA': 519}
    "ranger-f1-1220": 900,  # {'RS': 866, 'BO': 892, 'ASHA': 648}
    "ranger-f1-4154": 583,  # {'RS': 583, 'BO': 490, 'ASHA': 341}
    "ranger-f1-40978": 221,  # {'RS': 221, 'BO': 207, 'ASHA': 221}
    "ranger-f1-4534": 593,  # {'RS': 401, 'BO': 593, 'ASHA': 190}
    "ranger-f1-40900": 526,  # {'RS': 526, 'BO': 394, 'ASHA': 430}
    "ranger-f1-40536": 900,  # {'RS': 861, 'BO': 760, 'ASHA': 757}
    "ranger-f1-41156": 600,  # {'RS': 495, 'BO': 481, 'ASHA': 791}
    "ranger-f1-1590": 821,  # {'RS': 577, 'BO': 821, 'ASHA': 322}
    "ranger-f1-41157": 780,  # {'RS': 601, 'BO': 560, 'ASHA': 737}
    "ranger-f1-1461": 720,  # {'RS': 488, 'BO': 831, 'ASHA': 591}
    "ranger-f1-1464": 637,  # {'RS': 332, 'BO': 637, 'ASHA': 492}
    "ranger-f1-15": 264,  # {'RS': 110, 'BO': 264, 'ASHA': 92}
    "ranger-f1-41142": 900,  # {'RS': 775, 'BO': 842, 'ASHA': 866}
    "ranger-f1-40701": 580,  # {'RS': 444, 'BO': 553, 'ASHA': 580}
    "ranger-f1-40994": 557,  # {'RS': 557, 'BO': 326, 'ASHA': 500}
    "ranger-f1-29": 441,  # {'RS': 106, 'BO': 441, 'ASHA': 141}
    "ranger-f1-31": 626,  # {'RS': 296, 'BO': 626, 'ASHA': 614}
    "ranger-f1-6332": 384,  # {'RS': 384, 'BO': 109, 'ASHA': 365}
    "ranger-f1-37": 600,  # {'RS': 758, 'BO': 240, 'ASHA': 524}
    "ranger-f1-23381": 900,  # {'RS': 573, 'BO': 869, 'ASHA': 677}
    "ranger-f1-151": 900,  # {'RS': 741, 'BO': 890, 'ASHA': 416}
    "ranger-f1-1479": 780,  # {'RS': 716, 'BO': 346, 'ASHA': 707}
    "ranger-f1-1480": 780,  # {'RS': 542, 'BO': 850, 'ASHA': 741}
    "ranger-f1-41143": 600,  # {'RS': 362, 'BO': 739, 'ASHA': 704}
    "ranger-f1-1053": 900,  # {'RS': 811, 'BO': 862, 'ASHA': 889}
    "ranger-f1-1067": 900,  # {'RS': 853, 'BO': 498, 'ASHA': 868}
    "ranger-f1-1063": 600,  # {'RS': 867, 'BO': 789, 'ASHA': 835}
    "ranger-f1-3": 197,  # {'RS': 146, 'BO': 136, 'ASHA': 197}
    "ranger-f1-1485": 720,  # {'RS': 331, 'BO': 699, 'ASHA': 855}
    "ranger-f1-1056": 717,  # {'RS': 254, 'BO': 596, 'ASHA': 717}
    "ranger-f1-334": 600,  # {'RS': 562, 'BO': 194, 'ASHA': 846}
    "ranger-f1-1486": 584,  # {'RS': 415, 'BO': 538, 'ASHA': 584}
    "ranger-f1-1487": 780,  # {'RS': 791, 'BO': 397, 'ASHA': 497}
    "ranger-f1-1068": 720,  # {'RS': 807, 'BO': 475, 'ASHA': 875}
    "ranger-f1-1050": 900,  # {'RS': 531, 'BO': 885, 'ASHA': 863}
    "ranger-f1-1049": 511,  # {'RS': 511, 'BO': 177, 'ASHA': 504}
    "ranger-f1-1489": 588,  # {'RS': 588, 'BO': 552, 'ASHA': 527}
    "ranger-f1-470": 900,  # {'RS': 676, 'BO': 643, 'ASHA': 877}
    "ranger-f1-1494": 600,  # {'RS': 728, 'BO': 359, 'ASHA': 428}
    "ranger-f1-312": 540,  # {'RS': 362, 'BO': 621, 'ASHA': 766}
    "ranger-f1-38": 716,  # {'RS': 716, 'BO': 407, 'ASHA': 308}
    "ranger-f1-44": 720,  # {'RS': 449, 'BO': 777, 'ASHA': 818}
    "ranger-f1-1040": 311,  # {'RS': 311, 'BO': 205, 'ASHA': 308}
    "ranger-f1-41146": 690,  # {'RS': 586, 'BO': 356, 'ASHA': 690}
    "ranger-f1-40983": 481,  # {'RS': 481, 'BO': 127, 'ASHA': 406}
    "ranger-f1-41138": 720,  # {'RS': 684, 'BO': 727, 'ASHA': 610}
    "ranger-f1-1111": 886,  # {'RS': 138, 'BO': 226, 'ASHA': 886}
    "ranger-f1-41159": 900,  # {'RS': 844, 'BO': 858, 'ASHA': 722}
    "ranger-f1-41162": 713,  # {'RS': 53, 'BO': 248, 'ASHA': 713}
    "ranger-f1-23517": 618,  # {'RS': 618, 'BO': 609, 'ASHA': 609}
    "ranger-f1-41161": 239,  # {'RS': 239, 'BO': 68, 'ASHA': 134}
    "ranger-f1-41150": 900,  # {'RS': 842, 'BO': 634, 'ASHA': 571}
    "ranger-f1-23512": 474,  # {'RS': 410, 'BO': 474, 'ASHA': 464}
    "ranger-auc-4135": 560,  # {'RS': 420, 'BO': 472, 'ASHA': 560}
    "ranger-auc-40981": 341,  # {'RS': 282, 'BO': 86, 'ASHA': 341}
    "ranger-auc-4134": 720,  # {'RS': 696, 'BO': 650, 'ASHA': 753}
    "ranger-auc-1220": 720,  # {'RS': 385, 'BO': 864, 'ASHA': 724}
    "ranger-auc-4154": 491,  # {'RS': 427, 'BO': 468, 'ASHA': 491}
    "ranger-auc-4538": 753,  # {'RS': 596, 'BO': 708, 'ASHA': 753}
    "ranger-auc-40978": 549,  # {'RS': 549, 'BO': 308, 'ASHA': 549}
    "ranger-auc-375": 767,  # {'RS': 767, 'BO': 552, 'ASHA': 626}
    "ranger-auc-4534": 574,  # {'RS': 149, 'BO': 574, 'ASHA': 103}
    "ranger-auc-40900": 382,  # {'RS': 202, 'BO': 382, 'ASHA': 340}
    "ranger-auc-40536": 715,  # {'RS': 704, 'BO': 715, 'ASHA': 678}
    "ranger-auc-41156": 393,  # {'RS': 160, 'BO': 160, 'ASHA': 393}
    "ranger-auc-1590": 720,  # {'RS': 570, 'BO': 894, 'ASHA': 238}
    "ranger-auc-469": 180,  # {'RS': 281, 'BO': 288, 'ASHA': 276}
    "ranger-auc-41157": 653,  # {'RS': 498, 'BO': 515, 'ASHA': 653}
    "ranger-auc-1461": 900,  # {'RS': 576, 'BO': 633, 'ASHA': 869}
    "ranger-auc-1464": 547,  # {'RS': 490, 'BO': 547, 'ASHA': 211}
    "ranger-auc-41142": 900,  # {'RS': 746, 'BO': 860, 'ASHA': 770}
    "ranger-auc-40701": 291,  # {'RS': 160, 'BO': 252, 'ASHA': 291}
    "ranger-auc-23": 336,  # {'RS': 157, 'BO': 336, 'ASHA': 224}
    "ranger-auc-40668": 743,  # {'RS': 400, 'BO': 743, 'ASHA': 338}
    "ranger-auc-29": 491,  # {'RS': 285, 'BO': 240, 'ASHA': 491}
    "ranger-auc-31": 900,  # {'RS': 422, 'BO': 819, 'ASHA': 610}
    "ranger-auc-6332": 600,  # {'RS': 815, 'BO': 328, 'ASHA': 476}
    "ranger-auc-37": 357,  # {'RS': 145, 'BO': 357, 'ASHA': 272}
    "ranger-auc-23381": 550,  # {'RS': 550, 'BO': 531, 'ASHA': 289}
    "ranger-auc-151": 600,  # {'RS': 749, 'BO': 380, 'ASHA': 379}
    "ranger-auc-41164": 780,  # {'RS': 788, 'BO': 704, 'ASHA': 754}
    "ranger-auc-1475": 600,  # {'RS': 108, 'BO': 741, 'ASHA': 211}
    "ranger-auc-1478": 617,  # {'RS': 520, 'BO': 617, 'ASHA': 318}
    "ranger-auc-1479": 840,  # {'RS': 545, 'BO': 786, 'ASHA': 720}
    "ranger-auc-41212": 600,  # {'RS': 368, 'BO': 488, 'ASHA': 734}
    "ranger-auc-1480": 368,  # {'RS': 321, 'BO': 368, 'ASHA': 256}
    "ranger-auc-41143": 750,  # {'RS': 318, 'BO': 750, 'ASHA': 642}
    "ranger-auc-1053": 719,  # {'RS': 638, 'BO': 719, 'ASHA': 419}
    "ranger-auc-41027": 731,  # {'RS': 626, 'BO': 731, 'ASHA': 676}
    "ranger-auc-1067": 652,  # {'RS': 652, 'BO': 323, 'ASHA': 515}
    "ranger-auc-1063": 422,  # {'RS': 422, 'BO': 411, 'ASHA': 318}
    "ranger-auc-6": 340,  # {'RS': 249, 'BO': 336, 'ASHA': 340}
    "ranger-auc-1485": 686,  # {'RS': 686, 'BO': 561, 'ASHA': 667}
    "ranger-auc-1056": 600,  # {'RS': 818, 'BO': 220, 'ASHA': 406}
    "ranger-auc-14": 317,  # {'RS': 70, 'BO': 317, 'ASHA': 116}
    "ranger-auc-22": 704,  # {'RS': 80, 'BO': 704, 'ASHA': 124}
    "ranger-auc-1515": 294,  # {'RS': 286, 'BO': 250, 'ASHA': 294}
    "ranger-auc-334": 498,  # {'RS': 498, 'BO': 43, 'ASHA': 409}
    "ranger-auc-1486": 748,  # {'RS': 748, 'BO': 197, 'ASHA': 729}
    "ranger-auc-41278": 444,  # {'RS': 290, 'BO': 444, 'ASHA': 222}
    "ranger-auc-1487": 357,  # {'RS': 357, 'BO': 178, 'ASHA': 240}
    "ranger-auc-1068": 331,  # {'RS': 141, 'BO': 215, 'ASHA': 331}
    "ranger-auc-1050": 573,  # {'RS': 108, 'BO': 573, 'ASHA': 506}
    "ranger-auc-1049": 290,  # {'RS': 290, 'BO': 99, 'ASHA': 103}
    "ranger-auc-470": 466,  # {'RS': 466, 'BO': 321, 'ASHA': 410}
    "ranger-auc-1494": 242,  # {'RS': 75, 'BO': 111, 'ASHA': 242}
    "ranger-auc-182": 640,  # {'RS': 640, 'BO': 444, 'ASHA': 314}
    "ranger-auc-312": 120,  # {'RS': 72, 'BO': 101, 'ASHA': 103}
    "ranger-auc-44": 276,  # {'RS': 191, 'BO': 276, 'ASHA': 216}
    "ranger-auc-41146": 438,  # {'RS': 385, 'BO': 270, 'ASHA': 438}
    "ranger-auc-40499": 673,  # {'RS': 673, 'BO': 377, 'ASHA': 501}
    "ranger-auc-54": 684,  # {'RS': 35, 'BO': 684, 'ASHA': 32}
    "ranger-auc-41216": 427,  # {'RS': 399, 'BO': 121, 'ASHA': 427}
    "ranger-auc-307": 271,  # {'RS': 271, 'BO': 104, 'ASHA': 212}
    "ranger-auc-60": 600,  # {'RS': 426, 'BO': 795, 'ASHA': 260}
    "ranger-auc-40498": 652,  # {'RS': 360, 'BO': 570, 'ASHA': 652}
    "ranger-auc-181": 400,  # {'RS': 176, 'BO': 181, 'ASHA': 400}
    "ranger-auc-41163": 755,  # {'RS': 755, 'BO': 263, 'ASHA': 755}
    "ranger-auc-300": 80,  # {'RS': 56, 'BO': 12, 'ASHA': 74}
    "ranger-auc-23517": 900,  # {'RS': 869, 'BO': 815, 'ASHA': 245}
    "ranger-auc-41166": 900,  # {'RS': 815, 'BO': 768, 'ASHA': 691}
    "ranger-auc-23512": 677,  # {'RS': 677, 'BO': 558, 'ASHA': 593}
    "ranger-auc-41168": 900,  # {'RS': 761, 'BO': 787, 'ASHA': 820}
    "ranger-auc-1493": 291,  # {'RS': 49, 'BO': 105, 'ASHA': 291}
    "xgboost-f1-41143": 395,  # {'RS': 395, 'BO': 165, 'ASHA': 267}
    "xgboost-f1-470": 126,  # {'RS': 16, 'BO': 126, 'ASHA': 22}
    "xgboost-f1-1487": 80,  # {'RS': 30, 'BO': 30, 'ASHA': 80}
    "xgboost-f1-1461": 265,  # {'RS': 220, 'BO': 19, 'ASHA': 265}
    "xgboost-f1-31": 605,  # {'RS': 605, 'BO': 576, 'ASHA': 368}
    "xgboost-f1-1067": 147,  # {'RS': 51, 'BO': 147, 'ASHA': 93}
    "xgboost-f1-1590": 600,  # {'RS': 502, 'BO': 642, 'ASHA': 502}
    "xgboost-f1-40983": 111,  # {'RS': 89, 'BO': 36, 'ASHA': 111}
    "xgboost-f1-41163": 583,  # {'RS': 128, 'BO': 583, 'ASHA': 75}
    "xgboost-f1-1220": 351,  # {'RS': 208, 'BO': 351, 'ASHA': 241}
    "xgboost-f1-41159": 417,  # {'RS': 417, 'BO': 184, 'ASHA': 417}
    "xgboost-f1-1480": 55,  # {'RS': 13, 'BO': 55, 'ASHA': 12}
    "xgboost-f1-6332": 900,  # {'RS': 801, 'BO': 572, 'ASHA': 778}
    "xgboost-f1-1479": 900,  # {'RS': 604, 'BO': 145, 'ASHA': 829}
    "xgboost-f1-40536": 57,  # {'RS': 55, 'BO': 56, 'ASHA': 57}
    "xgboost-f1-41138": 840,  # {'RS': 730, 'BO': 735, 'ASHA': 210}
    "xgboost-f1-29": 900,  # {'RS': 590, 'BO': 536, 'ASHA': 823}
    "xgboost-f1-1462": 431,  # {'RS': 134, 'BO': 431, 'ASHA': 105}
    "xgboost-f1-1494": 562,  # {'RS': 562, 'BO': 471, 'ASHA': 254}
    "xgboost-f1-40701": 461,  # {'RS': 266, 'BO': 289, 'ASHA': 461}
    "xgboost-f1-44": 377,  # {'RS': 205, 'BO': 377, 'ASHA': 212}
    "xgboost-f1-334": 420,  # {'RS': 270, 'BO': 324, 'ASHA': 683}
    "xgboost-f1-41142": 600,  # {'RS': 500, 'BO': 745, 'ASHA': 347}
    "xgboost-f1-38": 129,  # {'RS': 129, 'BO': 43, 'ASHA': 91}
    "xgboost-f1-1050": 82,  # {'RS': 15, 'BO': 82, 'ASHA': 43}
    "xgboost-f1-23381": 80,  # {'RS': 80, 'BO': 80, 'ASHA': 75}
    "xgboost-f1-41157": 900,  # {'RS': 781, 'BO': 465, 'ASHA': 808}
    "xgboost-f1-15": 392,  # {'RS': 392, 'BO': 161, 'ASHA': 62}
    "xgboost-f1-4134": 720,  # {'RS': 678, 'BO': 856, 'ASHA': 297}
    "xgboost-f1-40981": 900,  # {'RS': 835, 'BO': 339, 'ASHA': 728}
    "xgboost-f1-41156": 314,  # {'RS': 215, 'BO': 215, 'ASHA': 314}
    "xgboost-f1-3": 563,  # {'RS': 205, 'BO': 547, 'ASHA': 563}
    "xgboost-f1-1049": 503,  # {'RS': 90, 'BO': 503, 'ASHA': 80}
    "xgboost-f1-1063": 568,  # {'RS': 568, 'BO': 201, 'ASHA': 36}
    "xgboost-f1-23512": 550,  # {'RS': 550, 'BO': 87, 'ASHA': 550}
    "xgboost-f1-1068": 88,  # {'RS': 12, 'BO': 88, 'ASHA': 12}
    "xgboost-f1-41161": 680,  # {'RS': 643, 'BO': 680, 'ASHA': 599}
    "xgboost-f1-1489": 600,  # {'RS': 770, 'BO': 86, 'ASHA': 450}
    "xgboost-f1-24": 132,  # {'RS': 132, 'BO': 78, 'ASHA': 132}
    "xgboost-f1-23517": 720,  # {'RS': 795, 'BO': 73, 'ASHA': 795}
    "xgboost-f1-1053": 693,  # {'RS': 19, 'BO': 693, 'ASHA': 330}
    "xgboost-f1-312": 351,  # {'RS': 301, 'BO': 269, 'ASHA': 351}
    "xgboost-f1-1510": 508,  # {'RS': 240, 'BO': 161, 'ASHA': 508}
    "xgboost-f1-41162": 239,  # {'RS': 239, 'BO': 33, 'ASHA': 239}
    "xgboost-f1-1464": 420,  # {'RS': 816, 'BO': 230, 'ASHA': 226}
    "xgboost-f1-41146": 535,  # {'RS': 535, 'BO': 372, 'ASHA': 231}
    "xgboost-f1-60": 561,  # {'RS': 44, 'BO': 11, 'ASHA': 561}
    "xgboost-f1-41150": 600,  # {'RS': 751, 'BO': 717, 'ASHA': 677}
    "xgboost-f1-37": 900,  # {'RS': 784, 'BO': 94, 'ASHA': 33}
    "xgboost-f1-1485": 900,  # {'RS': 878, 'BO': 605, 'ASHA': 500}
    "xgboost-f1-4534": 595,  # {'RS': 265, 'BO': 40, 'ASHA': 595}
    "xgboost-f1-151": 600,  # {'RS': 638, 'BO': 331, 'ASHA': 638}
    "xgboost-f1-40978": 900,  # {'RS': 555, 'BO': 758, 'ASHA': 666}
    "xgboost-f1-40994": 900,  # {'RS': 726, 'BO': 828, 'ASHA': 431}
    "xgboost-f1-50": 720,  # {'RS': 485, 'BO': 295, 'ASHA': 608}
    "xgboost-f1-1486": 600,  # {'RS': 325, 'BO': 604, 'ASHA': 385}
    "svm-f1-40981": 300,  # {'RS': 20, 'BO': 239, 'ASHA': 20}
    "svm-f1-40978": 780,  # {'RS': 175, 'BO': 137, 'ASHA': 781}
    "svm-f1-41138": 900,  # {'RS': 836, 'BO': 712, 'ASHA': 840}
    "svm-f1-41142": 480,  # {'RS': 392, 'BO': 364, 'ASHA': 761}
    "svm-f1-29": 780,  # {'RS': 661, 'BO': 494, 'ASHA': 363}
    "svm-f1-37": 80,  # {'RS': 35, 'BO': 9, 'ASHA': 45}
    "svm-f1-41143": 660,  # {'RS': 506, 'BO': 645, 'ASHA': 397}
    "svm-f1-24": 300,  # {'RS': 255, 'BO': 104, 'ASHA': 243}
    "svm-f1-41146": 480,  # {'RS': 145, 'BO': 46, 'ASHA': 421}
    "svm-f1-1485": 300,  # {'RS': 25, 'BO': 190, 'ASHA': 25}
    "svm-f1-1486": 360,  # {'RS': 15, 'BO': 300, 'ASHA': 15}
    "aknn-f1-41138": 759,  # {'RS': 291, 'BO': 759, 'ASHA': 702}
    "aknn-f1-40981": 360,  # {'RS': 100, 'BO': 106, 'ASHA': 728}
    "aknn-f1-4134": 720,  # {'RS': 848, 'BO': 254, 'ASHA': 206}
    "aknn-f1-1220": 900,  # {'RS': 879, 'BO': 383, 'ASHA': 796}
    "aknn-f1-4154": 720,  # {'RS': 709, 'BO': 765, 'ASHA': 609}
    "aknn-f1-40978": 600,  # {'RS': 642, 'BO': 344, 'ASHA': 348}
    "aknn-f1-1111": 900,  # {'RS': 367, 'BO': 885, 'ASHA': 462}
    "aknn-f1-41150": 404,  # {'RS': 166, 'BO': 378, 'ASHA': 404}
    "aknn-f1-4534": 660,  # {'RS': 398, 'BO': 666, 'ASHA': 520}
    "aknn-f1-40900": 510,  # {'RS': 237, 'BO': 117, 'ASHA': 510}
    "aknn-f1-40536": 568,  # {'RS': 471, 'BO': 437, 'ASHA': 568}
    "aknn-f1-41156": 594,  # {'RS': 383, 'BO': 118, 'ASHA': 594}
    "aknn-f1-1590": 550,  # {'RS': 550, 'BO': 400, 'ASHA': 222}
    "aknn-f1-41157": 780,  # {'RS': 567, 'BO': 489, 'ASHA': 740}
    "aknn-f1-1461": 720,  # {'RS': 400, 'BO': 824, 'ASHA': 666}
    "aknn-f1-1464": 626,  # {'RS': 551, 'BO': 107, 'ASHA': 626}
    "aknn-f1-15": 126,  # {'RS': 126, 'BO': 30, 'ASHA': 60}
    "aknn-f1-41142": 660,  # {'RS': 536, 'BO': 651, 'ASHA': 576}
    "aknn-f1-40701": 900,  # {'RS': 883, 'BO': 674, 'ASHA': 828}
    "aknn-f1-40994": 561,  # {'RS': 561, 'BO': 72, 'ASHA': 561}
    "aknn-f1-29": 176,  # {'RS': 41, 'BO': 42, 'ASHA': 176}
    "aknn-f1-31": 900,  # {'RS': 841, 'BO': 587, 'ASHA': 841}
    "aknn-f1-6332": 656,  # {'RS': 656, 'BO': 104, 'ASHA': 656}
    "aknn-f1-37": 472,  # {'RS': 472, 'BO': 142, 'ASHA': 421}
    "aknn-f1-23381": 396,  # {'RS': 396, 'BO': 78, 'ASHA': 382}
    "aknn-f1-151": 660,  # {'RS': 676, 'BO': 431, 'ASHA': 583}
    "aknn-f1-41159": 840,  # {'RS': 267, 'BO': 693, 'ASHA': 645}
    "aknn-f1-23512": 230,  # {'RS': 230, 'BO': 72, 'ASHA': 177}
    "aknn-f1-1480": 900,  # {'RS': 857, 'BO': 338, 'ASHA': 845}
    "aknn-f1-41143": 632,  # {'RS': 632, 'BO': 469, 'ASHA': 221}
    "aknn-f1-1053": 900,  # {'RS': 892, 'BO': 115, 'ASHA': 699}
    "aknn-f1-1067": 900,  # {'RS': 851, 'BO': 672, 'ASHA': 793}
    "aknn-f1-1063": 660,  # {'RS': 721, 'BO': 579, 'ASHA': 541}
    "aknn-f1-41162": 511,  # {'RS': 268, 'BO': 511, 'ASHA': 357}
    "aknn-f1-3": 161,  # {'RS': 161, 'BO': 67, 'ASHA': 123}
    "aknn-f1-1056": 429,  # {'RS': 429, 'BO': 238, 'ASHA': 325}
    "aknn-f1-334": 601,  # {'RS': 106, 'BO': 37, 'ASHA': 601}
    "aknn-f1-1486": 900,  # {'RS': 744, 'BO': 811, 'ASHA': 703}
    "aknn-f1-23517": 603,  # {'RS': 603, 'BO': 284, 'ASHA': 233}
    "aknn-f1-1487": 780,  # {'RS': 582, 'BO': 456, 'ASHA': 737}
    "aknn-f1-1068": 660,  # {'RS': 701, 'BO': 555, 'ASHA': 701}
    "aknn-f1-1050": 720,  # {'RS': 623, 'BO': 248, 'ASHA': 600}
    "aknn-f1-1049": 720,  # {'RS': 666, 'BO': 328, 'ASHA': 782}
    "aknn-f1-1489": 720,  # {'RS': 666, 'BO': 31, 'ASHA': 657}
    "aknn-f1-470": 371,  # {'RS': 371, 'BO': 161, 'ASHA': 371}
    "aknn-f1-1494": 600,  # {'RS': 541, 'BO': 96, 'ASHA': 741}
    "aknn-f1-41161": 900,  # {'RS': 302, 'BO': 839, 'ASHA': 640}
    "aknn-f1-312": 122,  # {'RS': 91, 'BO': 117, 'ASHA': 122}
    "aknn-f1-38": 900,  # {'RS': 847, 'BO': 311, 'ASHA': 723}
    "aknn-f1-44": 452,  # {'RS': 184, 'BO': 114, 'ASHA': 452}
    "aknn-f1-1040": 540,  # {'RS': 298, 'BO': 305, 'ASHA': 769}
    "aknn-f1-41146": 297,  # {'RS': 297, 'BO': 180, 'ASHA': 297}
    "aknn-f1-40983": 660,  # {'RS': 451, 'BO': 109, 'ASHA': 737}
    "aknn-auc-41138": 469,  # {'RS': 429, 'BO': 469, 'ASHA': 437}
    "aknn-auc-40981": 447,  # {'RS': 301, 'BO': 68, 'ASHA': 447}
    "aknn-auc-4134": 780,  # {'RS': 850, 'BO': 539, 'ASHA': 699}
    "aknn-auc-40927": 840,  # {'RS': 646, 'BO': 790, 'ASHA': 707}
    "aknn-auc-1220": 660,  # {'RS': 573, 'BO': 479, 'ASHA': 888}
    "aknn-auc-4154": 600,  # {'RS': 497, 'BO': 118, 'ASHA': 786}
    "aknn-auc-41163": 388,  # {'RS': 388, 'BO': 298, 'ASHA': 347}
    "aknn-auc-40996": 760,  # {'RS': 179, 'BO': 760, 'ASHA': 639}
    "aknn-auc-4538": 450,  # {'RS': 450, 'BO': 110, 'ASHA': 445}
    "aknn-auc-40978": 360,  # {'RS': 815, 'BO': 455, 'ASHA': 197}
    "aknn-auc-375": 58,  # {'RS': 25, 'BO': 58, 'ASHA': 24}
    "aknn-auc-1111": 660,  # {'RS': 217, 'BO': 545, 'ASHA': 755}
    "aknn-auc-40496": 131,  # {'RS': 100, 'BO': 36, 'ASHA': 131}
    "aknn-auc-41150": 654,  # {'RS': 654, 'BO': 368, 'ASHA': 566}
    "aknn-auc-4534": 600,  # {'RS': 802, 'BO': 222, 'ASHA': 647}
    "aknn-auc-40900": 244,  # {'RS': 82, 'BO': 87, 'ASHA': 244}
    "aknn-auc-40536": 840,  # {'RS': 706, 'BO': 42, 'ASHA': 781}
    "aknn-auc-41156": 677,  # {'RS': 677, 'BO': 339, 'ASHA': 594}
    "aknn-auc-1590": 660,  # {'RS': 886, 'BO': 384, 'ASHA': 588}
    "aknn-auc-1457": 585,  # {'RS': 585, 'BO': 356, 'ASHA': 338}
    "aknn-auc-469": 660,  # {'RS': 837, 'BO': 98, 'ASHA': 552}
    "aknn-auc-41157": 660,  # {'RS': 751, 'BO': 178, 'ASHA': 695}
    "aknn-auc-11": 120,  # {'RS': 121, 'BO': 61, 'ASHA': 176}
    "aknn-auc-1461": 600,  # {'RS': 875, 'BO': 176, 'ASHA': 847}
    "aknn-auc-1464": 757,  # {'RS': 521, 'BO': 36, 'ASHA': 757}
    "aknn-auc-15": 51,  # {'RS': 51, 'BO': 12, 'ASHA': 51}
    "aknn-auc-40975": 511,  # {'RS': 291, 'BO': 152, 'ASHA': 511}
    "aknn-auc-41142": 660,  # {'RS': 622, 'BO': 635, 'ASHA': 790}
    "aknn-auc-40701": 600,  # {'RS': 657, 'BO': 144, 'ASHA': 733}
    "aknn-auc-40994": 396,  # {'RS': 161, 'BO': 29, 'ASHA': 396}
    "aknn-auc-23": 426,  # {'RS': 426, 'BO': 53, 'ASHA': 402}
    "aknn-auc-40668": 660,  # {'RS': 738, 'BO': 392, 'ASHA': 745}
    "aknn-auc-29": 491,  # {'RS': 381, 'BO': 66, 'ASHA': 491}
    "aknn-auc-31": 900,  # {'RS': 826, 'BO': 784, 'ASHA': 102}
    "aknn-auc-6332": 621,  # {'RS': 312, 'BO': 42, 'ASHA': 621}
    "aknn-auc-37": 513,  # {'RS': 421, 'BO': 513, 'ASHA': 396}
    "aknn-auc-4541": 900,  # {'RS': 900, 'BO': 688, 'ASHA': 816}
    "aknn-auc-40670": 687,  # {'RS': 527, 'BO': 429, 'ASHA': 687}
    "aknn-auc-23381": 596,  # {'RS': 596, 'BO': 243, 'ASHA': 46}
    "aknn-auc-151": 400,  # {'RS': 260, 'BO': 400, 'ASHA': 358}
    "aknn-auc-188": 411,  # {'RS': 411, 'BO': 62, 'ASHA': 232}
    "aknn-auc-41164": 336,  # {'RS': 111, 'BO': 159, 'ASHA': 336}
    "aknn-auc-1475": 545,  # {'RS': 132, 'BO': 341, 'ASHA': 545}
    "aknn-auc-41159": 900,  # {'RS': 270, 'BO': 679, 'ASHA': 855}
    "aknn-auc-1478": 588,  # {'RS': 405, 'BO': 178, 'ASHA': 588}
    "aknn-auc-41169": 600,  # {'RS': 787, 'BO': 222, 'ASHA': 495}
    "aknn-auc-23512": 660,  # {'RS': 508, 'BO': 259, 'ASHA': 849}
    "aknn-auc-41212": 343,  # {'RS': 196, 'BO': 237, 'ASHA': 343}
    "aknn-auc-1480": 442,  # {'RS': 387, 'BO': 94, 'ASHA': 442}
    "aknn-auc-300": 334,  # {'RS': 203, 'BO': 157, 'ASHA': 334}
    "aknn-auc-41168": 780,  # {'RS': 726, 'BO': 259, 'ASHA': 714}
    "aknn-auc-41143": 662,  # {'RS': 341, 'BO': 618, 'ASHA': 662}
    "aknn-auc-1053": 612,  # {'RS': 612, 'BO': 163, 'ASHA': 589}
    "aknn-auc-41027": 526,  # {'RS': 526, 'BO': 115, 'ASHA': 283}
    "aknn-auc-1067": 512,  # {'RS': 336, 'BO': 68, 'ASHA': 512}
    "aknn-auc-1063": 240,  # {'RS': 210, 'BO': 78, 'ASHA': 210}
    "aknn-auc-41162": 699,  # {'RS': 397, 'BO': 677, 'ASHA': 699}
    "aknn-auc-3": 120,  # {'RS': 136, 'BO': 42, 'ASHA': 237}
    "aknn-auc-6": 146,  # {'RS': 146, 'BO': 29, 'ASHA': 91}
    "aknn-auc-1485": 447,  # {'RS': 447, 'BO': 52, 'ASHA': 329}
    "aknn-auc-1056": 668,  # {'RS': 650, 'BO': 111, 'ASHA': 668}
    "aknn-auc-14": 227,  # {'RS': 71, 'BO': 90, 'ASHA': 227}
    "aknn-auc-18": 600,  # {'RS': 316, 'BO': 103, 'ASHA': 828}
    "aknn-auc-22": 136,  # {'RS': 136, 'BO': 79, 'ASHA': 136}
    "aknn-auc-1515": 101,  # {'RS': 101, 'BO': 29, 'ASHA': 44}
    "aknn-auc-554": 373,  # {'RS': 171, 'BO': 373, 'ASHA': 181}
    "aknn-auc-1486": 649,  # {'RS': 550, 'BO': 310, 'ASHA': 649}
    "aknn-auc-23517": 780,  # {'RS': 229, 'BO': 131, 'ASHA': 717}
    "aknn-auc-41278": 692,  # {'RS': 606, 'BO': 643, 'ASHA': 692}
    "aknn-auc-1493": 388,  # {'RS': 219, 'BO': 62, 'ASHA': 388}
    "aknn-auc-1487": 617,  # {'RS': 617, 'BO': 36, 'ASHA': 487}
    "aknn-auc-1068": 720,  # {'RS': 847, 'BO': 590, 'ASHA': 847}
    "aknn-auc-1050": 600,  # {'RS': 311, 'BO': 31, 'ASHA': 810}
    "aknn-auc-1049": 546,  # {'RS': 546, 'BO': 137, 'ASHA': 51}
    "aknn-auc-1489": 671,  # {'RS': 671, 'BO': 316, 'ASHA': 257}
    "aknn-auc-470": 631,  # {'RS': 607, 'BO': 210, 'ASHA': 631}
    "aknn-auc-1494": 506,  # {'RS': 501, 'BO': 101, 'ASHA': 506}
    "aknn-auc-41161": 660,  # {'RS': 288, 'BO': 757, 'ASHA': 337}
    "aknn-auc-41165": 900,  # {'RS': 301, 'BO': 829, 'ASHA': 875}
    "aknn-auc-182": 211,  # {'RS': 211, 'BO': 39, 'ASHA': 123}
    "aknn-auc-312": 216,  # {'RS': 51, 'BO': 19, 'ASHA': 216}
    "aknn-auc-40984": 60,  # {'RS': 81, 'BO': 18, 'ASHA': 86}
    "aknn-auc-1501": 119,  # {'RS': 119, 'BO': 43, 'ASHA': 42}
    "aknn-auc-38": 637,  # {'RS': 637, 'BO': 30, 'ASHA': 619}
    "aknn-auc-44": 471,  # {'RS': 471, 'BO': 128, 'ASHA': 444}
    "aknn-auc-46": 198,  # {'RS': 198, 'BO': 80, 'ASHA': 148}
    "aknn-auc-40982": 286,  # {'RS': 286, 'BO': 112, 'ASHA': 282}
    "aknn-auc-1040": 468,  # {'RS': 468, 'BO': 146, 'ASHA': 228}
    "aknn-auc-41146": 361,  # {'RS': 256, 'BO': 162, 'ASHA': 361}
    "aknn-auc-54": 532,  # {'RS': 186, 'BO': 56, 'ASHA': 532}
    "aknn-auc-41216": 660,  # {'RS': 863, 'BO': 218, 'ASHA': 592}
    "aknn-auc-41166": 551,  # {'RS': 551, 'BO': 327, 'ASHA': 120}
    "aknn-auc-1497": 600,  # {'RS': 276, 'BO': 329, 'ASHA': 814}
    "aknn-auc-60": 660,  # {'RS': 542, 'BO': 869, 'ASHA': 519}
    "aknn-auc-40983": 147,  # {'RS': 147, 'BO': 24, 'ASHA': 147}
    "aknn-auc-40498": 628,  # {'RS': 628, 'BO': 305, 'ASHA': 263}
    "aknn-auc-181": 600,  # {'RS': 562, 'BO': 428, 'ASHA': 727}
    "aknn-auc-40923": 496,  # {'RS': 418, 'BO': 496, 'ASHA': 327}
}


def yahpo_rbv2_benchmark_definitions(method, restrict_fidelities=False) -> dict:
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
        for name in yahpo_rbv2_instances[method][metric]
    }


def yahpo_rbv2_selected_benchmark_definitions(
    method, restrict_fidelities=False
) -> dict:
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
        for name in yahpo_rbv2_selected_instances[method][metric]
    }
