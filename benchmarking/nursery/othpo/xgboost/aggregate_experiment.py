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
import os
import pickle
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils import store_json_res

exp_dir = Path(__file__).parent.parent / "xgboost_experiment_results"


exp_folder = str(exp_dir) + "/random-mnist/"

"""
Aggregate the results from multiple sagemaker experiments testing different hyperparameter values.
"""

exp_files = os.listdir(exp_folder)

experiments = []
for exp_ff in exp_files:
    if exp_ff[:11] == "XGBoost_HPO":
        experiments.append(pickle.load(open(exp_folder + exp_ff, "rb")))

tot_num_hps = len(experiments[0]["parameters_mat"]["learning_rates"])
num_data_sizes = len(experiments[0]["data_sizes"])

agg_experiments = copy.deepcopy(experiments[0])
agg_experiments["num_hyp_pars"] = tot_num_hps
agg_experiments["hyp_id_start"] = None
agg_experiments["hyp_id_end"] = None

agg_experiments["test_error_mat"] = np.ones((num_data_sizes, tot_num_hps, 1)) * np.nan
agg_experiments["train_error_mat"] = np.ones((num_data_sizes, tot_num_hps, 1)) * np.nan
agg_experiments["execution_times"] = np.ones((num_data_sizes, tot_num_hps, 1)) * np.nan


for exp in experiments:
    start_idx = exp["hyp_id_start"]
    end_idx = exp["hyp_id_end"]
    agg_experiments["test_error_mat"][:, start_idx : end_idx + 1, :] = exp[
        "test_error_mat"
    ]
    agg_experiments["train_error_mat"][:, start_idx : end_idx + 1, :] = exp[
        "train_error_mat"
    ]
    agg_experiments["execution_times"][:, start_idx : end_idx + 1, :] = exp[
        "execution_times"
    ]

for key in ["test_error_mat", "train_error_mat", "execution_times"]:
    assert np.sum(np.isnan(agg_experiments[key])) == 0

store_json_res(agg_experiments, exp_folder + "aggregated_experiments.json")
