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
import pytest
import pandas as pd
import numpy as np
import itertools
import sys

from backend_definitions_dict import BACKEND_DEFS


# Dataframes for tests
metric, _, _, _ = BACKEND_DEFS["SimOpt"]
points_per_task = 3

df_0 = pd.DataFrame(
    {
        "price_A": [1, 3, 5],
        "price_B": [2, 6, 7],
        "price_C": [3, 6, 9],
        "time_idx": [0, 0, 0],
        "status": ["Completed"] * points_per_task,
        metric: [10, 20, 30],
    }
)

df_1 = pd.DataFrame(
    {
        "price_A": [1, 3, 5],
        "price_B": [2, 6, 7],
        "price_C": [3, 6, 9],
        "time_idx": [1, 1, 1],
        "status": ["Completed"] * points_per_task,
        metric: [8, 10, 12],
    }
)

df_2 = pd.DataFrame(
    {
        "price_A": [1, 3, 5],
        "price_B": [2, 6, 7],
        "price_C": [3, 6, 9],
        "time_idx": [2, 2, 2],
        "status": ["Completed"] * points_per_task,
        metric: [12, 10, 28],
    }
)

df_3 = pd.DataFrame(
    {
        "price_A": [1, 3, 5],
        "price_B": [2, 6, 7],
        "price_C": [3, 6, 9],
        "time_idx": [3, 3, 3],
        "status": ["Completed"] * points_per_task,
        metric: [12, 30, 30],
    }
)

df_4 = pd.DataFrame(
    {
        "price_A": [1, 2, 3],
        "price_B": [1, 2, 3],
        "price_C": [1, 2, 3],
        "time_idx": [3, 3, 3],
        "status": ["Completed"] * points_per_task,
        metric: [2, 10, -10],
    }
)

df_5 = pd.DataFrame(
    {
        "price_A": [10, 20, 30],
        "price_B": [10, 20, 30],
        "price_C": [10, 20, 30],
        "time_idx": [3, 3, 3],
        "status": ["Completed"] * points_per_task,
        metric: [0, 1, 2],
    }
)


@pytest.mark.skip()
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="BoTorch requires python 3.8 or higher"
)
@pytest.mark.timeout(10)
def test_num_optima():
    from blackbox_helper import (
        get_transfer_points_active,
        simopt_backend_conf,
        num_optima,
    )

    backend_file = "simopt/SimOptNewsPrice.py"
    getbackend_simopt = lambda active_task_val: simopt_backend_conf(
        backend_file, active_task_val
    )

    metric, _, active_task_str, _ = BACKEND_DEFS["SimOpt"]
    opt_mode = "min"  # Not the standard for SimOpt

    _, conf_space, _ = getbackend_simopt(0)

    assert 0 == num_optima(
        past_points=[],
        past_df=None,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check num_optima on zero inputs."

    assert 1 == num_optima(
        past_points=[],
        past_df=df_0,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check num_optima on no prev_points but past_df."

    # Add df_0 to past_points
    past_points = get_transfer_points_active(
        df_0, [0], points_per_task, conf_space, None, active_task_str, metric
    )

    assert 1 == num_optima(
        past_points=past_points,
        past_df=None,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check on one previous task."

    assert 1 == num_optima(
        past_points=past_points,
        past_df=df_1,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check on additional past_df when the minima are the same."

    # Add df_1 to past_points
    past_points = get_transfer_points_active(
        df_1, [0, 1], points_per_task, conf_space, past_points, active_task_str, metric
    )

    assert 1 == num_optima(
        past_points=past_points,
        past_df=None,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check on two tasks when the minima are the same."

    assert 2 == num_optima(
        past_points=past_points,
        past_df=df_2,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check on additional past_df when the minima are not the same."

    # Add df_2 to past_points
    past_points = get_transfer_points_active(
        df_2,
        [0, 1, 2],
        points_per_task,
        conf_space,
        past_points,
        active_task_str,
        metric,
    )

    assert 2 == num_optima(
        past_points=past_points,
        past_df=None,
        opt_mode=opt_mode,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check on three tasks when the minima are not the same."

    assert 1 == num_optima(
        past_points=past_points,
        past_df=None,
        opt_mode="max",
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check counting maxima."

    # Add df_3 to past_points
    past_points = get_transfer_points_active(
        df_3,
        [0, 1, 2, 3],
        points_per_task,
        conf_space,
        past_points,
        active_task_str,
        metric,
    )

    assert 1 == num_optima(
        past_points=past_points,
        past_df=None,
        opt_mode="max",
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        n_points=points_per_task,
    ), "Check when joint maxima."


simopt_backend_file = "simopt/SimOptNewsPrice.py"
xgboost_res_file = "xgboost_experiment_results/random-mnist/aggregated_experiments.json"
yahpo_dataset = "1220"
yahpo_scenario = "rbv2_svm"

backends = ["SimOpt", "YAHPO", "XGBoost"]
optimisers = [
    "BoundingBox",
    "ZeroShot",
    "Quantiles",
    "WarmBO",
    "WarmBOShuffled",
    "BoTorchTransfer",
    "BayesianOptimization",
    "RandomSearch",
]

optimiser_type_dict = {
    "BoundingBox": "Transfer",
    "ZeroShot": "Transfer",
    "Quantiles": "Transfer",
    "WarmBO": "Transfer",
    "WarmBOShuffled": "Transfer",
    "BoTorchTransfer": "Transfer",
    "BayesianOptimization": "Naive",
    "RandomSearch": "Naive",
}

combinations = list(itertools.product(backends, optimisers))


@pytest.mark.skip()
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="BoTorch requires python 3.8 or higher"
)
@pytest.mark.timeout(60)
@pytest.mark.parametrize("backend, optimiser", combinations)
def test_backends_transfer(backend, optimiser):
    from collect_results import collect_res

    optimiser_type = optimiser_type_dict[optimiser]
    seed = 2
    points_per_task = 10

    results = collect_res(
        seed_start=seed,
        seed_end=seed,
        timestamp=None,
        points_per_task=points_per_task,
        optimiser=optimiser,
        optimiser_type=optimiser_type,
        backend=backend,
        xgboost_res_file=xgboost_res_file,
        simopt_backend_file=simopt_backend_file,
        yahpo_dataset=yahpo_dataset,
        yahpo_scenario=yahpo_scenario,
        metric=None,
        store_res=False,
        task_lim=3,
    )

    res = results[(backend, optimiser, seed)]

    _, _, active_task_str, _ = BACKEND_DEFS[backend]
    for act_task in res:
        if active_task_str in res[act_task]:
            values_not_nan = res[act_task][active_task_str][
                ~np.isnan(res[act_task][active_task_str])
            ]

            assert (
                act_task == values_not_nan
            ).all(), "Check that active task choice is respected."


@pytest.mark.skip()
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="BoTorch requires python 3.8 or higher"
)
@pytest.mark.timeout(60)
@pytest.mark.parametrize("metric", ["auc", "acc"])
def test_different_metrics(metric):
    from collect_results import collect_res

    optimiser = "BayesianOptimization"
    optimiser_type = optimiser_type_dict[optimiser]
    backend = "YAHPO"
    yahpo_dataset = "1220"
    yahpo_scenario = "rbv2_svm"
    seed = 4
    points_per_task = 10

    results = collect_res(
        seed_start=seed,
        seed_end=seed,
        timestamp=None,
        points_per_task=points_per_task,
        optimiser=optimiser,
        optimiser_type=optimiser_type,
        backend=backend,
        xgboost_res_file=None,
        simopt_backend_file=None,
        yahpo_dataset=yahpo_dataset,
        yahpo_scenario=yahpo_scenario,
        metric=metric,
        store_res=False,
        task_lim=3,
    )

    res = results[(backend, optimiser, seed)]


#########
# Prepare for test_StudentBO
#########
ordered_sol_df5_v1 = pd.DataFrame(
    {
        "price_A": [30, 2, 5, 3, 20],
        "price_B": [30, 2, 7, 6, 20],
        "price_C": [30, 2, 9, 6, 20],
    }
)

ordered_sol_df5_v2 = pd.DataFrame(
    {
        "price_A": [30, 2, 3, 5, 20],
        "price_B": [30, 2, 6, 7, 20],
        "price_C": [30, 2, 6, 9, 20],
    }
)

ordered_sol_df3_v1 = pd.DataFrame(
    {"price_A": [5, 3, 1], "price_B": [7, 6, 2], "price_C": [9, 6, 3]}
)

ordered_sol_df3_v2 = pd.DataFrame(
    {"price_A": [3, 5, 1], "price_B": [6, 7, 2], "price_C": [6, 9, 3]}
)


StudentBO_Combs = [
    ("short", [df_0, df_1, df_2, df_3], [ordered_sol_df3_v1, ordered_sol_df3_v2]),
    (
        "long",
        [df_0, df_1, df_2, df_3, df_4, df_5],
        [ordered_sol_df5_v1, ordered_sol_df5_v2],
    ),
]


@pytest.mark.skip()
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="BoTorch requires python 3.8 or higher"
)
@pytest.mark.parametrize("tst_id, dataframes, exp_sols", StudentBO_Combs)
def test_StudentBO(tst_id, dataframes, exp_sols):
    from blackbox_helper import (
        get_transfer_points_active,
        get_configs,
        initialise_scheduler_stopping_criterion,
    )

    metric, opt_mode, active_task_str, _ = BACKEND_DEFS["SimOpt"]

    _, getbackend = get_configs(
        "SimOpt",
        xgboost_res_file=None,
        simopt_backend_file=simopt_backend_file,
        yahpo_dataset=None,
    )
    _, conf_space, _ = getbackend(0)
    base_kwargs = {
        "config_space": conf_space,
        "mode": opt_mode,
        "metric": metric,
        "random_seed": 10,
    }

    # Build up TransferPoints
    past_points = None
    for df_idx in range(len(dataframes)):
        df = dataframes[df_idx]
        past_points = get_transfer_points_active(
            df,
            list(range(df_idx)),
            points_per_task,
            conf_space,
            past_points,
            active_task_str,
            metric,
        )

    transfer_kwargs = copy.deepcopy(base_kwargs)
    transfer_kwargs["transfer_learning_evaluations"] = past_points

    scheduler_sorted, _ = initialise_scheduler_stopping_criterion(
        "WarmBO",
        base_kwargs,
        transfer_kwargs,
        points_per_task,
        past_points,
        True,
        active_task_val=None,
    )

    points_ordered = pd.DataFrame(scheduler_sorted.searcher._points_to_evaluate)

    assert (exp_sols[0] == points_ordered).all().all() or (
        exp_sols[1] == points_ordered
    ).all().all()

    points_shuffled = []
    for _ in range(5):
        scheduler_shuffl, _ = initialise_scheduler_stopping_criterion(
            "WarmBOShuffled",
            base_kwargs,
            transfer_kwargs,
            points_per_task,
            past_points,
            True,
            active_task_val=None,
        )

        points_shuffled.append(
            pd.DataFrame(scheduler_shuffl.searcher._points_to_evaluate)
        )

    if tst_id == "short":
        # Enough shared optima that the points are the same even if we shuffle
        assert np.array(
            [
                (points_shuffled[pp] == exp_sols[0]).all().all()
                or (points_shuffled[pp] == exp_sols[1]).all().all()
                for pp in range(len(points_shuffled))
            ]
        ).all()
    else:
        assert not np.array(
            [
                (points_shuffled[pp] == points_ordered).all().all()
                for pp in range(len(points_shuffled))
            ]
        ).all()
