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
import pandas as pd
import copy

import syne_tune.config_space as sp
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository import (
    add_surrogate,
    BlackboxRepositoryBackend,
    UserBlackboxBackend,
)
from syne_tune.optimizer.baselines import BoTorch, RandomSearch, ZeroShotTransfer
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
    BoundingBox,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    QuantileBasedSurrogateSearcher,
)
from syne_tune.backend import LocalBackend

from bo_warm_transfer import WarmStartBayesianOptimization

from syne_tune.optimizer.schedulers.searchers.botorch.botorch_transfer_searcher import (
    BoTorchTransfer,
)

from utils import load_json_res

# Partly based on https://github.com/awslabs/syne-tune/blob/main/examples/launch_simulated_benchmark.py


def get_configs(
    backend: str,
    xgboost_res_file: str = None,
    simopt_backend_file: str = None,
    yahpo_dataset: str = None,
    yahpo_scenario: str = None,
):
    """
    Produce get_backend function required by do_tasks_in_order
    """
    if backend == "SimOpt":
        assert (
            simopt_backend_file is not None
        ), "SimOpt backend requires a problem file."
        task_list = list(range(0, 9))
        return task_list, lambda active_task_val: simopt_backend_conf(
            simopt_backend_file, active_task_val
        )
    if backend == "XGBoost":
        assert (
            xgboost_res_file is not None
        ), "XGBoost requires a file with experiment evaluations."
        res = load_json_res(xgboost_res_file)
        X, Y, E, order, data_sizes = massage_data(res)
        task_list = [int(dd) for dd in data_sizes]
        return task_list, lambda active_task_val: xgboost_backend_conf(
            X, Y, order, data_sizes, E, active_task_val
        )
    if backend == "YAHPO":
        assert yahpo_dataset is not None, "YAHPO requires specifying the dataset."
        assert yahpo_scenario is not None, "YAHPO requires specifying the scenario."
        print(
            "Using yahpo scenario %s and data set %s." % (yahpo_scenario, yahpo_dataset)
        )
        task_list = list(range(1, 21))
        return task_list, lambda active_task_val: yahpo_backend_conf(
            active_task_val, dataset=yahpo_dataset, scenario=yahpo_scenario
        )


def massage_data(res):

    N_hyp = len(res["parameters_mat"]["learning_rates"])

    order = sorted(res["parameters_mat"].keys())
    try:
        data_sizes = res["data_sizes"]
    except:
        data_sizes = np.array(
            [int(min(0.8 * 70000, 70000 * 10**ii)) for ii in np.linspace(-3, 0, 28)]
        )

    # Test error
    Y = np.reshape(np.array(res["test_error_mat"])[:, :, 0], -1)
    # Execution time
    E = np.reshape(np.array(res["execution_times"])[:, :, 0], -1)

    # Hyperparameters
    H_once = np.array([res["parameters_mat"][feat] for feat in order])
    H = np.tile(H_once, len(data_sizes))
    X = np.vstack([H, np.repeat(data_sizes, N_hyp)])

    return X, Y, E, order, data_sizes


def xgboost_blackbox(X, Y, order, data_sizes, E, data_size):
    # Set up blackbox
    hyperparameters = pd.DataFrame(data=X.T, columns=order + ["data_size"])
    config_space = {
        "learning_rates": sp.loguniform(1e-6, 1.0),
        "min_child_weight": sp.loguniform(1e-6, 32),
        "max_depth": sp.lograndint(2, 32),
        "n_estimators": sp.lograndint(2, 256),
    }

    # Only use evalations from one data_size
    idx = hyperparameters["data_size"] == data_size
    hyperparameters = hyperparameters[idx]
    del hyperparameters["data_size"]

    cs_fidelity = {
        "hp_epoch": sp.randint(1, 1),
    }

    Y_box = np.ones((len(Y[idx]), 1, 1, 2))
    Y_box[:, 0, 0, 0] = Y[idx]
    Y_box[:, 0, 0, 1] = E[idx]

    return add_surrogate(
        BlackboxTabular(
            hyperparameters=hyperparameters,
            configuration_space=config_space,
            fidelity_space=cs_fidelity,
            objectives_evaluations=Y_box,
            objectives_names=["metric_error", "runtime"],
        )
    )


def get_points_to_evaluate_myoptic(
    df, past_sizes, n_points, conf_space, past_points, active_task_str, metric
):
    return []


def get_first_n_df(raw_df, n_points, interested_keys):
    df = raw_df[raw_df["status"] == "Completed"].reset_index()
    return df[[key for key in interested_keys]][:n_points]


def get_transfer_points_active(
    raw_df, past_sizes, n_points, conf_space, tasks, active_task_str, metric
):
    # Make TransferLearningTaskEvaluations from the dataframe of the run on the previous active_task_val
    if tasks is None or tasks == []:
        tasks = {}
    if past_sizes == []:
        return []
    df = get_first_n_df(raw_df, n_points, list(conf_space.keys()) + [metric])
    pp = past_sizes[-1]
    evals = np.zeros((min(n_points, len(df)), 1, 1, 1))
    hps = df[[key for key in conf_space]]
    evals[:, 0, 0, 0] = list(df[metric][:n_points])

    tasks[pp] = TransferLearningTaskEvaluations(
        configuration_space=conf_space,
        hyperparameters=hps,
        objectives_evaluations=evals,
        objectives_names=[metric],
    )

    return tasks


def simopt_backend_conf(backend, active_task_val):

    conf_space = {
        "price_A": sp.randint(0, 20),
        "price_B": sp.randint(0, 20),
        "price_C": sp.randint(0, 20),
    }

    trial_backend = ActiveLocalBackend(
        entry_point=backend, active_task_val=active_task_val, active_task_str="time_idx"
    )

    callbacks = []
    return trial_backend, conf_space, callbacks


def xgboost_backend_conf(X, Y, order, active_task_list, E, active_task_val):
    blackbox = xgboost_blackbox(
        X, Y, order, data_sizes=active_task_list, E=E, data_size=active_task_val
    )

    trial_backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr="runtime",
    )
    callbacks = [SimulatorCallback()]
    return trial_backend, blackbox.configuration_space, callbacks


def yahpo_backend_conf(active_task_val, dataset="41156", scenario="rbv2_svm"):

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name="yahpo-%s" % scenario,
        elapsed_time_attr="timetrain",
        dataset=dataset,
        surrogate_kwargs={"fidelities": [active_task_val]},
    )
    callbacks = [SimulatorCallback()]
    return trial_backend, trial_backend.blackbox.configuration_space, callbacks


class ActiveLocalBackend(LocalBackend):
    def __init__(
        self,
        entry_point: str,
        delete_checkpoints: bool = False,
        rotate_gpus: bool = True,
        active_task_val: int = 0,
        active_task_str: str = None,
    ):
        self.active_task_val = active_task_val
        self.active_task_str = active_task_str
        super().__init__(entry_point, delete_checkpoints, rotate_gpus)

    def _schedule(self, trial_id: int, config: dict):
        config_wider = copy.deepcopy(config)
        config_wider[self.active_task_str] = self.active_task_val
        super()._schedule(trial_id, config_wider)


def _count_optima(list_best_configs_org):
    list_best_configs = copy.deepcopy(list_best_configs_org)

    # First pass: add all configs from sets with only one optima
    collected_optima = set({})
    ii = 0
    while ii < len(list_best_configs):
        if len(list_best_configs[ii]) == 1:
            cur_config = list_best_configs.pop(ii).pop()
            collected_optima.add(cur_config)
        else:
            ii += 1

    # Then keep adding configs till all lists are represented
    list_size = 2
    while len(list_best_configs) > 0:
        ii = 0
        while len(list_best_configs) > 0 and ii < len(list_best_configs):
            if len(list_best_configs[ii]) == list_size:
                cur_set = list_best_configs.pop(ii)
                if len(cur_set.intersection(collected_optima)) > 0:
                    pass
                    # This set is already represented
                else:
                    # Add one at random
                    # Note: this means it's not guaranteed to find smallest number if more than 2
                    collected_optima.add(cur_set.pop())
            else:
                ii += 1
        list_size += 1
    return len(collected_optima)


def _get_optima_idx(opt_mode, values):
    if opt_mode == "min":
        return np.where(values == np.min(values))[0]
    else:
        return np.where(values == np.max(values))[0]


def _make_config_str(keys_to_use, opt_idx, df):
    return str({key: df[key][opt_idx] for key in keys_to_use})


def _get_task_set(keys_to_use, hyp_df, metric_vals, opt_mode):
    return set(
        [
            _make_config_str(keys_to_use, opt_idx, hyp_df)
            for opt_idx in _get_optima_idx(opt_mode, metric_vals)
        ]
    )


def num_optima(
    past_points, active_task_str, conf_space, metric, opt_mode, n_points, past_df=None
):
    """
    Count the number of optima in past_points and past_df. Used for BoundingBox, which
    requires >= two different optima.
    """
    list_best_configs = []
    keys_to_use = [key for key in conf_space if key != active_task_str]

    if past_df is not None:
        # Get the top configs from past_df
        filt_past_df = get_first_n_df(
            past_df, n_points, list(conf_space.keys()) + [metric]
        )

        list_best_configs.append(
            _get_task_set(keys_to_use, filt_past_df, filt_past_df[metric], opt_mode)
        )

    if past_points is None or past_points == [] or past_points == {}:
        return len(list_best_configs)

    for task_id in past_points:
        # Get the top configs for each task_id
        _, D1, D2, D3 = np.shape(past_points[task_id].objectives_evaluations)
        assert D1 == D2 == D3 == 1, "We assume one-dimensional objectives"
        vals = past_points[task_id].objectives_evaluations[:, 0, 0, 0]
        list_best_configs.append(
            _get_task_set(
                keys_to_use, past_points[task_id].hyperparameters, vals, opt_mode
            )
        )

    return _count_optima(list_best_configs)


def initialise_scheduler_stopping_criterion(
    optimiser,
    base_kwargs,
    transfer_kwargs,
    points_per_task,
    past_points,
    check_enough_tasks,
    active_task_val,
):

    stop_criterion = StoppingCriterion(max_num_trials_completed=points_per_task)

    if optimiser == "ZeroShot" and check_enough_tasks:
        # Needs a constrained config_space in scheduler and blackbox
        scheduler = ZeroShotTransfer(
            sort_transfer_learning_evaluations=True,
            use_surrogates=True,
            **transfer_kwargs,
        )

    elif optimiser == "WarmBO" and check_enough_tasks:
        scheduler = WarmStartBayesianOptimization(
            num_warm_points=5,
            sort_by_task_id=True,
            **transfer_kwargs,
        )

    elif optimiser == "PrevBO" and check_enough_tasks:
        last_task_arg = np.argmax(list(past_points.keys()))
        last_task_key = list(past_points.keys())[last_task_arg]
        prev_points = {last_task_key: past_points[last_task_key]}
        transfer_kwargs["transfer_learning_evaluations"] = prev_points
        scheduler = WarmStartBayesianOptimization(
            num_warm_points=5,
            sort_by_task_id=True,
            **transfer_kwargs,
        )
    elif optimiser == "PrevNoBO" and check_enough_tasks:
        last_task_arg = np.argmax(list(past_points.keys()))
        last_task_key = list(past_points.keys())[last_task_arg]
        prev_points = {last_task_key: past_points[last_task_key]}
        transfer_kwargs["transfer_learning_evaluations"] = prev_points
        scheduler = WarmStartBayesianOptimization(
            num_warm_points=25,
            sort_by_task_id=True,
            **transfer_kwargs,
        )

    elif optimiser == "WarmBOShuffled" and check_enough_tasks:
        scheduler = WarmStartBayesianOptimization(
            num_warm_points=5,
            sort_by_task_id=True,
            shuffle_order=True,
            **transfer_kwargs,
        )

    elif optimiser == "BoTorchTransfer" and check_enough_tasks:
        scheduler = BoTorchTransfer(
            new_task_id=active_task_val,
            encode_tasks_ordinal=True,
            points_to_evaluate=[],
            **transfer_kwargs,
        )

    elif optimiser == "BoundingBox" and check_enough_tasks:
        del transfer_kwargs["random_seed"]
        del base_kwargs["config_space"]
        scheduler = BoundingBox(
            scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
                new_config_space,
                points_to_evaluate=[],
                searcher="botorch",
                **base_kwargs,
            ),
            **transfer_kwargs,
        )

    elif optimiser == "Quantiles" and check_enough_tasks:
        scheduler = FIFOScheduler(
            points_to_evaluate=[],
            searcher=QuantileBasedSurrogateSearcher(**transfer_kwargs),
            **base_kwargs,
        )

    elif optimiser == "RandomSearch":
        scheduler = RandomSearch(**base_kwargs, points_to_evaluate=[])

    else:  # BayesianOptimization
        if not check_enough_tasks:
            # BO is being used to warm up other method
            past_points_to_use = []  # Need this to be [] instead of None
            print("Using BO to warm up %s" % optimiser)
        else:
            past_points_to_use = past_points

        scheduler = BoTorch(
            points_to_evaluate=past_points_to_use,
            **base_kwargs,
        )

        stop_criterion = StoppingCriterion(
            max_num_trials_completed=len(past_points_to_use) + points_per_task
        )

    return scheduler, stop_criterion


def do_tasks_in_order(
    seed,
    active_task_list,
    pte_func,
    points_per_task,
    get_backend,
    optimiser="BayesianOptimization",
    metric="metric_error",
    opt_mode="min",
    active_task_str=None,
    uses_fidelity=False,
    n_workers=4,
):
    assert active_task_str is not None

    _, conf_space, _ = get_backend(active_task_val=active_task_list[0])

    num_optima_wr = lambda past_points, past_df=None: num_optima(
        past_points=past_points,
        active_task_str=active_task_str,
        conf_space=conf_space,
        metric=metric,
        opt_mode=opt_mode,
        n_points=points_per_task,
        past_df=past_df,
    )

    warm_up_its = {
        "ZeroShot": 1,
        "WarmBO": 1,
        "PrevBO": 1,
        "PrevNoBO": 1,
        "WarmBOShuffled": 1,
        "BoTorchTransfer": 1,
        "BoundingBox": 2,
        "Quantiles": 1,
    }

    past_task_vals = []
    results = {}
    past_df = None
    past_points = None
    for active_task_val in active_task_list:
        print("---------------------------------------------------------------")
        print("seed: %s \t active_task_val: %s" % (seed, active_task_val))

        print("Prev optimia: %s" % num_optima_wr(past_points, past_df))

        trial_backend, conf_space, callbacks = get_backend(
            active_task_val=active_task_val
        )

        past_points = pte_func(
            past_df,
            past_task_vals,
            points_per_task,
            conf_space,
            past_points,
            active_task_str,
            metric,
        )

        base_kwargs = {
            "config_space": conf_space,
            "mode": opt_mode,
            "metric": metric,
            "random_seed": seed,
        }
        transfer_kwargs = copy.deepcopy(base_kwargs)
        transfer_kwargs["transfer_learning_evaluations"] = past_points

        check_enough_tasks = (
            num_optima_wr(past_points) >= warm_up_its[optimiser]
            if optimiser in warm_up_its
            else True
        )

        scheduler, stop_criterion = initialise_scheduler_stopping_criterion(
            optimiser,
            base_kwargs,
            transfer_kwargs,
            points_per_task,
            past_points,
            check_enough_tasks,
            active_task_val,
        )

        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0,
            # This callback is required in order to make things work with the
            # simulator callback. It makes sure that results are stored with
            # simulated time (rather than real time), and that the time_keeper
            # is advanced properly whenever the tuner loop sleeps
            callbacks=callbacks,
            tuner_name=optimiser,
        )
        tuner.run()

        past_task_vals.append(active_task_val)
        past_df = tuner.tuning_status.get_dataframe()
        results[active_task_val] = past_df
    return results
