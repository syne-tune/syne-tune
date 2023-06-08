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
from blackbox_helper import (
    get_points_to_evaluate_myoptic,
    get_transfer_points_active,
    do_tasks_in_order,
    get_configs,
)

from backend_definitions_dict import BACKEND_DEFS

import pickle
import argparse
import os


def collect_res(
    timestamp,
    points_per_task,
    optimiser,
    optimiser_type,
    backend,
    seed_start=0,
    seed_end=0,
    xgboost_res_file=None,
    simopt_backend_file=None,
    yahpo_dataset=None,
    yahpo_scenario=None,
    metric=None,
    store_res=True,
    task_lim=None,
    run_locally=False,
):
    if optimiser_type == "Transfer":
        pte_func = get_transfer_points_active
    elif optimiser_type == "Naive":
        pte_func = get_points_to_evaluate_myoptic
    else:
        raise ValueError

    metric_def, opt_mode, active_task_str, uses_fidelity = BACKEND_DEFS[backend]
    full_task_list, get_backend = get_configs(
        backend, xgboost_res_file, simopt_backend_file, yahpo_dataset, yahpo_scenario
    )

    if metric is None:
        print("Using default metric %s" % metric_def)
        metric = metric_def
    else:
        print("Using given metric %s" % metric)

    if task_lim is None:
        active_task_list = full_task_list
    else:
        active_task_list = [int(aa) for aa in full_task_list[:task_lim]]

    results = {}
    for seed in range(seed_start, seed_end + 1):
        res = do_tasks_in_order(
            seed=seed,
            active_task_list=active_task_list,
            pte_func=pte_func,
            points_per_task=points_per_task,
            get_backend=get_backend,
            optimiser=optimiser,
            metric=metric,
            opt_mode=opt_mode,
            active_task_str=active_task_str,
            uses_fidelity=uses_fidelity,
            n_workers=4,
        )
        results[(backend, optimiser, seed)] = res
        if store_res:
            print("Storing result")
            if run_locally:
                print("Storing locally")
                result_folder = "optimisation_results"
                os.makedirs(result_folder, exist_ok=True)
            else:
                print("Storing remotely")
                result_folder = os.environ.get("SM_MODEL_DIR")
            pickle.dump(
                results, open(result_folder + "/collect_results_%s.p" % timestamp, "wb")
            )
            print(
                "Result stored at %s"
                % (result_folder + "/collect_results_%s.p" % timestamp)
            )
    return results


def get_parser():
    """
    Generates the parser for the different hyper parameters.
    """
    parser = argparse.ArgumentParser()

    # getting the hyper parameters:
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=0)
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--points_per_task", type=int)
    parser.add_argument("--optimiser", type=str)
    parser.add_argument("--optimiser_type", type=str)
    parser.add_argument("--backend", type=str)
    parser.add_argument("--xgboost_res_file", type=str, default=None)
    parser.add_argument("--simopt_backend_file", type=str, default=None)
    parser.add_argument("--yahpo_dataset", type=str, default=None)
    parser.add_argument("--yahpo_scenario", type=str, default=None)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--run_locally", type=bool, default=False)
    return parser


if __name__ == "__main__":

    parser = get_parser()

    args, _ = parser.parse_known_args()

    args_dict = vars(args)

    collect_res(**args_dict)
