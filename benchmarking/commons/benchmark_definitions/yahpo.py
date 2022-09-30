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
from benchmarking.commons.benchmark_definitions.common import BenchmarkDefinition


# Note: We do not include scenarios `iaml_super`, `rbv2_super` for now,
# because they need proper usage of conditional configuration spaces,
# which is not supported yet


# TODO:
# - Check that elapsed_time_attr is always cumulative
# - Check that metric is validation, what we want
# - Run 2 baseline methods with all these benchmarks, in order to figure out
#   the most interesting ones


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


def yahpo_nb301_benchmark(dataset_name):
    return BenchmarkDefinition(
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


def yahpo_lcbench_benchmark(dataset_name):
    return BenchmarkDefinition(
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


yahpo_fcnet_instances = {
    "naval": "fcnet_naval_propulsion",
    "protein": "fcnet_protein_structure",
    "slice": "fcnet_slice_localization",
    "parkinsons": "fcnet_parkinsons_telemonitoring",
}


def yahpo_fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=1200,
        n_workers=4,
        elapsed_time_attr="runtime",
        metric="valid_loss",
        mode="min",
        blackbox_name="yahpo-fcnet",
        dataset_name=yahpo_fcnet_instances[dataset_name],
        max_resource_attr="max_resources",
    )


yahpo_fcnet_benchmark_definitions = {
    "yahpo-fcnet-" + name: yahpo_fcnet_benchmark(name) for name in yahpo_fcnet_instances
}


def yahpo_iaml_rpart_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_rpart",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_iaml_instances = ["40981", "41146", "1489", "1067"]


yahpo_iaml_rpart_benchmark_definitions = {
    f"yahpo-iaml_rpart_{metric}-{name}": yahpo_iaml_rpart_benchmark(name, metric, mode)
    for name in yahpo_iaml_instances
    for metric, mode in (
        ("f1", "max"),
        ("auc", "max"),
    )
}


def yahpo_iaml_glmnet_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_glmnet",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_iaml_metrics = (("f1", "max"), ("auc", "max"))


yahpo_iaml_glmnet_benchmark_definitions = {
    f"yahpo-iaml_glmnet_{metric}-{name}": yahpo_iaml_glmnet_benchmark(
        name, metric, mode
    )
    for name in yahpo_iaml_instances
    for metric, mode in yahpo_iaml_metrics
}


def yahpo_iaml_ranger_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_ranger",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_iaml_ranger_benchmark_definitions = {
    f"yahpo-iaml_ranger_{metric}-{name}": yahpo_iaml_ranger_benchmark(
        name, metric, mode
    )
    for name in yahpo_iaml_instances
    for metric, mode in yahpo_iaml_metrics
}


def yahpo_iaml_xgboost_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-iaml_xgboost",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_iaml_xgboost_benchmark_definitions = {
    f"yahpo-iaml_xgboost_{metric}-{name}": yahpo_iaml_xgboost_benchmark(
        name, metric, mode
    )
    for name in yahpo_iaml_instances
    for metric, mode in yahpo_iaml_metrics
}


def yahpo_rbv2_rpart_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_rpart",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_instances = [
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
    "40966",
    "41150",
    "4534",
    "40900",
    "40536",
    "41156",
    "1590",
    "1457",
    "458",
    "469",
    "41157",
    "11",
    "1461",
    "1462",
    "1464",
    "15",
    "40975",
    "41142",
    "40701",
    "40994",
    "23",
    "1468",
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
    "3",
    "6",
    "1485",
    "1056",
    "12",
    "14",
    "16",
    "18",
    "40979",
    "22",
    "1515",
    "554",
    "334",
    "24",
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
    "40984",
    "1501",
    "40685",
    "38",
    "42",
    "44",
    "46",
    "40982",
    "1040",
    "41146",
    "377",
    "40499",
    "50",
    "54",
    "41166",
    "307",
    "1497",
    "60",
    "1510",
    "40983",
    "40498",
    "181",
]


yahpo_rbv2_metrics = (("acc", "max"), ("f1", "max"), ("auc", "max"))

yahpo_rbv2_rpart_benchmark_definitions = {
    f"yahpo-rbv2_rpart_{metric}-{name}": yahpo_rbv2_rpart_benchmark(name, metric, mode)
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}


def yahpo_rbv2_glmnet_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_glmnet",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_glmnet_benchmark_definitions = {
    f"yahpo-rbv2_glmnet_{metric}-{name}": yahpo_rbv2_glmnet_benchmark(
        name, metric, mode
    )
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}


def yahpo_rbv2_ranger_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_ranger",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_ranger_benchmark_definitions = {
    f"yahpo-rbv2_ranger_{metric}-{name}": yahpo_rbv2_ranger_benchmark(
        name, metric, mode
    )
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}


def yahpo_rbv2_xgboost_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_xgboost",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_xgboost_benchmark_definitions = {
    f"yahpo-rbv2_xgboost_{metric}-{name}": yahpo_rbv2_xgboost_benchmark(
        name, metric, mode
    )
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}


def yahpo_rbv2_svm_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_svm",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_svm_benchmark_definitions = {
    f"yahpo-rbv2_svm_{metric}-{name}": yahpo_rbv2_svm_benchmark(name, metric, mode)
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}


def yahpo_rbv2_aknn_benchmark(dataset_name, metric, mode="max"):
    return BenchmarkDefinition(
        max_wallclock_time=1800,  # TODO
        n_workers=4,
        elapsed_time_attr="timetrain",
        metric=metric,
        mode=mode,
        blackbox_name="yahpo-rbv2_aknn",
        dataset_name=dataset_name,
        max_resource_attr="max_resources",
    )


yahpo_rbv2_aknn_benchmark_definitions = {
    f"yahpo-rbv2_aknn_{metric}-{name}": yahpo_rbv2_aknn_benchmark(name, metric, mode)
    for name in yahpo_rbv2_instances
    for metric, mode in yahpo_rbv2_metrics
}
