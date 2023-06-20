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
import numpy as np
import os
from pathlib import Path

cur_folder = str(Path(__file__).parent)
output_folder = cur_folder + "/Figures"
os.makedirs(output_folder, exist_ok=True)

legend_order = {
    "WarmBO": 7,
    "WarmBOShuffled": 6,
    "BoTorchTransfer": 5,
    "Quantiles": 4,
    "ZeroShot": 3,
    "BoundingBox": 2,
    "BayesianOptimization": 1,
    "RandomSearch": 0,
    "PrevBO": 8,
    "PrevNoBO": 9,
}

colours = {
    "BoundingBox": "maroon",
    "ZeroShot": "chocolate",
    "Quantiles": "orange",
    "BayesianOptimization": "darkseagreen",
    "WarmBO": "navy",
    "BoTorchTransfer": "mediumorchid",
    "RandomSearch": "forestgreen",
    "WarmBOShuffled": "cornflowerblue",
    "PrevBO": "palevioletred",
    "PrevNoBO": "black",
}

hatches = {
    "BoundingBox": "//",
    "ZeroShot": "//",
    "Quantiles": "//",
    "BayesianOptimization": "",
    "WarmBO": "--",
    "BoTorchTransfer": "--",
    "RandomSearch": "",
    "WarmBOShuffled": "--",
    "PrevBO": "--",
    "PrevNoBO": "--",
}

linestyles = {
    "BoundingBox": "-.",
    "ZeroShot": "-.",
    "Quantiles": "-.",
    "BayesianOptimization": "-",
    "WarmBO": "--",
    "BoTorchTransfer": "--",
    "RandomSearch": "-",
    "WarmBOShuffled": "--",
    "PrevBO": "--",
    "PrevNoBO": "--",
}

labels = {
    "BoundingBox": "BoundingBox",
    "ZeroShot": "ZeroShot",
    "Quantiles": "CTS",
    "BayesianOptimization": "BO",
    "WarmBO": "SimpleOrdered",
    "BoTorchTransfer": "TransferBO",
    "RandomSearch": "RandomSearch",
    "WarmBOShuffled": "SimpleOrderedShuffled",
    "PrevBO": "SimplePrevious",
    "PrevNoBO": "SimplePreviousNoBO",
}


def task_pos_in_order(task_val, task_values):
    return np.where(task_val == np.array(task_values))[0][0] + 1


def sort_legend_labels(label_list, handles):
    leg_order = [legend_order[method] for method in label_list]
    ordered_handles = [hh for _, hh in sorted(zip(leg_order, handles))]
    ordered_labels = [labels[ll] for _, ll in sorted(zip(leg_order, label_list))]
    return ordered_handles, ordered_labels


def get_task_values_to_plot(backend, task_values):
    if backend == "SimOpt":
        task_values_to_plot = [1, 2, 4, 6, 8]
    elif backend == "YAHPO":
        task_values_to_plot = [2, 3, 5, 10, 20]
    elif backend == "XGBoost":
        task_values_to_plot = np.array(task_values)[[1, 7, 14, 21, 27]]
    else:
        raise ValueError
    return task_values_to_plot
