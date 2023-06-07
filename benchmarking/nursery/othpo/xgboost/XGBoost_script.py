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
import XGBoost_helper

import argparse
import numpy as np
import pickle
import time
import os
from sklearn import datasets


def evaluate_XGBoost(
    hyp_dict_file, hyp_id_start, hyp_id_end, timestamp, dataset, run_locally=False
):
    print(timestamp)

    print("Starting XGBoost part")

    if dataset == "digits":
        dataset = datasets.load_digits()
        size_lin_space = np.linspace(-1, 0, 10)
    elif dataset == "mnist":
        dataset = datasets.fetch_openml("mnist_784")
        size_lin_space = np.linspace(-3, 0, 28)
    else:
        raise ValueError

    X = np.array(dataset.data)
    Y = np.array(dataset.target).astype("uint32")
    test_threshold = int(0.8 * len(Y))

    data_sizes = np.array([int(test_threshold * 10**ii) for ii in size_lin_space])
    num_hyp_pars = hyp_id_end - hyp_id_start + 1

    model_name, llik = "XGBClassifier", False
    print(data_sizes)

    num_seeds = 1
    split_seed = 1

    X_train, Y_train, X_test, Y_test = XGBoost_helper.get_data_splits(
        split_seed, X, Y, test_threshold, smallest_train=np.min(data_sizes)
    )

    parameters_mat = pickle.load(open(hyp_dict_file, "rb"))

    train_error_mat = []
    test_error_mat = []
    execution_times = []
    for ii in range(len(data_sizes)):
        train_error_mat.append([])
        test_error_mat.append([])
        execution_times.append([])
        for jj in range(num_hyp_pars):
            train_error_mat[ii].append([])
            test_error_mat[ii].append([])
            execution_times[ii].append([])

    to_store = {
        "parameters_mat": parameters_mat,
        "train_error_mat": train_error_mat,
        "test_error_mat": test_error_mat,
        "execution_times": execution_times,
        "data_sizes": data_sizes,
        "hyp_dict_file": hyp_dict_file,
        "num_hyp_pars": num_hyp_pars,
        "hyp_id_start": hyp_id_start,
        "hyp_id_end": hyp_id_end,
    }

    for seed in list(range(num_seeds)):
        print(seed)

        for ii in range(len(data_sizes)):
            data_s = data_sizes[ii]
            for jj in range(num_hyp_pars):
                print(data_s, jj)

                start_t = time.time()

                par_dict = {
                    "learning_rate": parameters_mat["learning_rates"][
                        jj + hyp_id_start
                    ],
                    "max_depth": parameters_mat["max_depth"][jj + hyp_id_start],
                    "min_child_weight": parameters_mat["min_child_weight"][
                        jj + hyp_id_start
                    ],
                    "n_estimators": parameters_mat["n_estimators"][jj + hyp_id_start],
                }

                test_error, train_error, _, _ = XGBoost_helper.train_and_evaluate(
                    X_train[:data_s],
                    Y_train[:data_s],
                    par_dict,
                    X_test,
                    Y_test,
                    model_name,
                    llik,
                )

                to_store["train_error_mat"][ii][jj].append(train_error)
                to_store["test_error_mat"][ii][jj].append(test_error)
                to_store["execution_times"][ii][jj].append(time.time() - start_t)

                print(train_error, test_error, to_store["execution_times"][ii][jj])

                if run_locally:
                    output_folder = "../xgboost_experiment_results/random-mnist"
                    os.makedirs(output_folder, exist_ok=True)
                else:
                    output_folder = os.environ.get("SM_MODEL_DIR")
                pickle.dump(
                    to_store,
                    open(output_folder + "/XGBoost_HPO_%s.p" % timestamp, "wb"),
                )

        print("Completed!")


def get_parser():
    """
    Generates the parser for the different hyper parameters.
    """
    parser = argparse.ArgumentParser()

    # getting the hyper parameters:
    parser.add_argument("--hyp_dict_file", type=str)
    parser.add_argument("--hyp_id_start", type=int, default=0)
    parser.add_argument("--hyp_id_end", type=int, default=999)
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--run_locally", type=bool, default=False)
    return parser


if __name__ == "__main__":
    import os

    parser = get_parser()

    args, _ = parser.parse_known_args()

    args_dict = vars(args)

    evaluate_XGBoost(**args_dict)
