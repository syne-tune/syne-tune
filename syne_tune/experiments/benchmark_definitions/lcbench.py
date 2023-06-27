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
from syne_tune.experiments.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)


def lcbench_benchmark(dataset_name: str, datasets=None) -> SurrogateBenchmarkDefinition:
    """
    The default is to use nearest neighbour regression with ``K=1``. If
    you use a more sophisticated surrogate, it is recommended to also
    define ``add_surrogate_kwargs``, for example:

    .. code-block:: python

       surrogate="RandomForestRegressor",
       add_surrogate_kwargs={
           "predict_curves": True,
           "fit_differences": ["time"],
       },

    :param dataset_name: Value for ``dataset_name``
    :param datasets: Used for transfer learning
    :return: Definition of benchmark
    """
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",  # 1-nn surrogate
        surrogate_kwargs={"n_neighbors": 1},
        max_num_evaluations=4000,
        datasets=datasets,
        max_resource_attr="epochs",
    )


lcbench_datasets = [
    "KDDCup09_appetency",
    "covertype",
    "Amazon_employee_access",
    "adult",
    "nomao",
    "bank-marketing",
    "shuttle",
    "Australian",
    "kr-vs-kp",
    "mfeat-factors",
    "credit-g",
    "vehicle",
    "kc1",
    "blood-transfusion-service-center",
    "cnae-9",
    "phoneme",
    "higgs",
    "connect-4",
    "helena",
    "jannis",
    "volkert",
    "MiniBooNE",
    "APSFailure",
    "christine",
    "fabert",
    "airlines",
    "jasmine",
    "sylvine",
    "albert",
    "dionis",
    "car",
    "segment",
    "Fashion-MNIST",
    "jungle_chess_2pcs_raw_endgame_complete",
]

lcbench_benchmark_definitions = {
    "lcbench-" + task: lcbench_benchmark(task) for task in lcbench_datasets
}


# 5 most expensive lcbench datasets
lcbench_selected_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]

lcbench_selected_benchmark_definitions = {
    "lcbench-"
    + task.replace("_", "-").replace(".", ""): lcbench_benchmark(
        task, datasets=lcbench_selected_datasets
    )
    for task in lcbench_selected_datasets
}
