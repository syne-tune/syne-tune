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
import argparse
import copy
import datetime
import numpy as np
import os
import pickle

from backend_definitions_dict import BACKEND_DEFS
from blackbox_helper import get_configs
from experiment_master_file import experiments_meta_dict


def preprocess_res(**kwargs):

    backend = kwargs["backend"]

    task_values, _ = get_configs(
        backend=backend,
        xgboost_res_file=kwargs["xgboost_res_file"],
        simopt_backend_file=kwargs["simopt_backend_file"],
        yahpo_dataset=kwargs["yahpo_dataset"],
        yahpo_scenario=kwargs["yahpo_scenario"],
    )

    metric_def, opt_mode, _, _ = BACKEND_DEFS[backend]
    if kwargs["metric"] is None:
        metric = metric_def
        kwargs["metric"] = metric_def
    else:
        metric = kwargs["metric"]

    # Load result files for chosen experiment
    res_files = []
    for ff in kwargs["files"]:
        res_files.append(pickle.load(open("optimisation_results/" + ff, "rb")))

    # Extract and organise the runs, calculating best value seen yet for each iteration
    evaluation_lines = {}
    methods = []
    for res_ff in res_files:
        res_keys = res_ff.keys()
        # Two different keys in different versions
        for key in res_keys:
            if len(key) == 2:
                method, seed = key
            else:
                backend_check, method, seed = key
                assert backend == backend_check

            # Initialise dict
            if method not in evaluation_lines:
                evaluation_lines[method] = {val: [] for val in task_values}
                methods.append(method)

            for task_val in task_values:
                df = res_ff[key][task_val]
                scores = df[df["status"] == "Completed"][metric]
                if opt_mode == "max":
                    line = [
                        np.max(scores.iloc[: ii + 1]) for ii in range(points_per_task)
                    ]
                else:
                    line = [
                        np.min(scores.iloc[: ii + 1]) for ii in range(points_per_task)
                    ]
                evaluation_lines[method][task_val].append(line)

    # Calculate mean performance accross seeds
    mean_performance = {method: {} for method in methods}
    std_mean_performance = {method: {} for method in methods}
    for method in methods:
        for task_val in task_values:
            lines = evaluation_lines[method][task_val]
            mean_performance[method][task_val] = np.mean(lines, 0)
            std_mean_performance[method][task_val] = np.std(lines, 0, ddof=1) / np.sqrt(
                len(lines)
            )

    # Used for metric calculations
    max_means = {val: -np.inf for val in task_values}
    min_means = {val: np.inf for val in task_values}
    for method in methods:
        for task_val in task_values:
            max_means[task_val] = max(
                max_means[task_val], np.max(mean_performance[method][task_val])
            )
            min_means[task_val] = min(
                min_means[task_val], np.min(mean_performance[method][task_val])
            )
    if opt_mode == "max":
        best_means = max_means
        worst_means = min_means
    else:
        best_means = min_means
        worst_means = max_means

    # Metric calculations
    projected_means = {}
    for method in methods:
        projected_means[method] = {}
        for task_val in task_values:
            proj = (
                worst_means[task_val] - np.array(evaluation_lines[method][task_val])
            ) / (worst_means[task_val] - best_means[task_val])
            projected_means[method][task_val] = np.mean(proj, 0)

    # Calculating normalised scores
    mean_normalised_score = {method: {} for method in methods}
    std_mean_normalised_score = {method: {} for method in methods}
    for method in methods:
        for task_val in task_values:
            normalised_scores = (
                100
                * (evaluation_lines[method][task_val] - best_means[task_val])
                / (
                    mean_performance["RandomSearch"][task_val][points_per_task - 1]
                    - best_means[task_val]
                )
            )

            mean_normalised_score[method][task_val] = np.mean(normalised_scores, 0)
            std_mean_normalised_score[method][task_val] = np.std(
                normalised_scores, 0, ddof=1
            ) / np.sqrt(len(normalised_scores))

    # Number of iterations needed to reach fraction of best score
    mean_its_qq = {}
    for method in methods:
        mean_its_qq[method] = {}
        for task_val in task_values:
            mean_its_qq[method][task_val] = {
                qq: np.argmax(projected_means[method][task_val] >= qq)
                for qq in qq_values
            }
            # Check whether the fraction was reached at all and if not fix
            for qq in qq_values:
                if projected_means[method][task_val][-1] < qq:
                    mean_its_qq[method][task_val][qq] = points_per_task

    preprocessed_results = copy.deepcopy(kwargs)
    preprocessed_results["evaluation_lines"] = evaluation_lines
    preprocessed_results["methods"] = methods
    preprocessed_results["mean_performance"] = mean_performance
    preprocessed_results["std_mean_performance"] = std_mean_performance
    preprocessed_results["best_means"] = best_means
    preprocessed_results["worst_means"] = worst_means
    preprocessed_results["projected_means"] = projected_means
    preprocessed_results["mean_normalised_score"] = mean_normalised_score
    preprocessed_results["std_mean_normalised_score"] = std_mean_normalised_score
    preprocessed_results["mean_its_qq"] = mean_its_qq
    preprocessed_results["task_values"] = task_values
    preprocessed_results["preprocessing_timestamp"] = datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    os.makedirs("plotting", exist_ok=True)
    pickle.dump(
        preprocessed_results,
        open("plotting/preprocessed_results_%s.p" % kwargs["experiment"], "wb"),
    )
    print("Preprocessing completed")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", type=str, help="What experiment to preprocess the results of"
    )
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    print("Preprocessing results for %s" % args.experiment)
    points_per_task = 25
    qq_values = [0.75, 0.9, 0.95]

    preprocess_res(
        **experiments_meta_dict[args.experiment],
        experiment=args.experiment,
        points_per_task=points_per_task,
        qq_values=qq_values
    )
