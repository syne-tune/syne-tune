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
import copy
import numpy as np

from typing import Dict, Any

from syne_tune.optimizer.baselines import BoTorch
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)


def _make_config_str(keys_to_use, opt_idx, df, de_str_dict):
    config = {key: df[key][opt_idx] for key in keys_to_use}
    conf_str = str(config)
    de_str_dict[conf_str] = config
    return conf_str


def de_string_config(config_str, de_str_dict):
    return de_str_dict[config_str]


def get_optima_but_nn_idx(opt_mode, values, nn):
    if opt_mode == "min":
        ordered_values = sorted(np.unique(values))
    else:
        ordered_values = list(reversed(sorted(np.unique(values))))
    try:
        cur_val = ordered_values[nn]
        return np.where(values == cur_val)[0]
    except:
        return []


def return_optima_from_task(
    task: TransferLearningTaskEvaluations,
    opt_mode: str,
    keys_to_use: list,
    metric: str,
    nn: int,
    de_str_dict,
):
    optima_idx = get_optima_but_nn_idx(
        opt_mode, task.objective_values(metric)[:, 0, 0], nn
    )
    if len(optima_idx) == 0:
        return None, None
    opt_val = task.objective_values(metric)[optima_idx[0], 0, 0]
    return opt_val, set(
        [
            _make_config_str(keys_to_use, opt_idx, task.hyperparameters, de_str_dict)
            for opt_idx in optima_idx
        ]
    )


def get_optima_all_tasks(
    transfer_learning_evaluations, opt_mode, keys_to_use, metric, nn, de_str_dict
):
    all_optima = []
    for task_id in transfer_learning_evaluations:
        opt_val, opt_configs = return_optima_from_task(
            transfer_learning_evaluations[task_id],
            opt_mode,
            keys_to_use,
            metric,
            nn,
            de_str_dict,
        )
        if opt_val is not None:
            all_optima.append((task_id, opt_val, opt_configs))
    return all_optima


def get_sorted_optima(
    transfer_learning_evaluations,
    opt_mode,
    keys_to_use,
    metric,
    nn,
    de_str_dict,
    sort_by_task_id,
    shuffle_order,
):

    optimal_points = copy.deepcopy(
        get_optima_all_tasks(
            transfer_learning_evaluations,
            opt_mode,
            keys_to_use,
            metric,
            nn=nn,
            de_str_dict=de_str_dict,
        )
    )

    if sort_by_task_id:
        # Sort so greater task_id comes first
        optimal_points.sort(key=lambda tup: -tup[0])
    else:
        sign = -1 if opt_mode == "max" else 1
        optimal_points.sort(key=lambda tup: sign * tup[1])

    if shuffle_order:
        np.random.shuffle(optimal_points)

    return optimal_points


def choose_warm_starters(
    num_warm,
    transfer_learning_evaluations,
    opt_mode,
    keys_to_use,
    metric,
    sort_by_task_id,
    shuffle_order,
):
    points_to_evaluate = []
    nn = 0

    de_str_dict = {}

    optimal_points = get_sorted_optima(
        transfer_learning_evaluations,
        opt_mode,
        keys_to_use,
        metric,
        nn,
        de_str_dict,
        sort_by_task_id,
        shuffle_order,
    )

    while len(optimal_points) > 0:
        while len(points_to_evaluate) < num_warm and len(optimal_points) > 0:
            task_to_add = optimal_points.pop(0)
            config_to_add = task_to_add[2].pop()
            if config_to_add not in points_to_evaluate:
                points_to_evaluate.append(config_to_add)
            if len(task_to_add[2]) > 0:  # still joint optima left
                optimal_points.append(task_to_add)
        nn += 1
        if len(points_to_evaluate) == num_warm:
            break

        optimal_points = get_sorted_optima(
            transfer_learning_evaluations,
            opt_mode,
            keys_to_use,
            metric,
            nn,
            de_str_dict,
            sort_by_task_id,
            shuffle_order,
        )

    return [de_string_config(point, de_str_dict) for point in points_to_evaluate]


class WarmStartBayesianOptimization(BoTorch):
    """
    Bayesian optimization where we use the previous tasks in transfer_learning_evaluations to decide the first points to sample,
    hence warm-starting the method.
    The choice of num_warm_points needs to balance the amount we learn from previous tasks with the amount of task-specific
    exploration.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        transfer_learning_evaluations: Dict[Any, TransferLearningTaskEvaluations],
        num_warm_points: int,
        points_to_evaluate=[],
        sort_by_task_id=None,
        shuffle_order=False,
        **kwargs
    ):

        assert len(transfer_learning_evaluations) >= 1, "need a task to transfer from"

        if sort_by_task_id:
            assert np.array(
                [
                    type(key) in [int, float]
                    for key in transfer_learning_evaluations.keys()
                ]
            ).all(), "task ids should be numeric if we're sorting by them."

        super(WarmStartBayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=[],
            **kwargs,
        )
        self.transfer_learning_evaluations = transfer_learning_evaluations
        self.num_warm_points = num_warm_points
        warm_points = choose_warm_starters(
            num_warm_points,
            transfer_learning_evaluations,
            kwargs["mode"],
            config_space.keys(),
            metric,
            sort_by_task_id,
            shuffle_order,
        )
        if len(warm_points) < num_warm_points:
            print("Adding %s random points" % (num_warm_points - len(warm_points)))
            self.searcher.num_minimum_observations += num_warm_points - len(warm_points)
        if points_to_evaluate is not None:
            self.searcher._points_to_evaluate += points_to_evaluate
        self.searcher._points_to_evaluate += warm_points
