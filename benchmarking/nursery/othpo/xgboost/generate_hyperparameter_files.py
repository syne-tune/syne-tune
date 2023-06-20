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
import pickle
import numpy as np

gen_seed = 13
num_hyp_pars = 1000

np.random.seed(gen_seed)

# Random ####
learning_rates = [
    XGBoost_helper.make_sample(
        samp_type="continuous", min_lim=1e-6, max_lim=1, scale="log"
    )
    for _ in range(num_hyp_pars)
]

min_child_weight = [
    XGBoost_helper.make_sample(
        samp_type="continuous", min_lim=1e-6, max_lim=32, scale="log"
    )
    for _ in range(num_hyp_pars)
]

max_depth = [
    XGBoost_helper.make_sample(samp_type="integer", min_lim=2, max_lim=32, scale="log")
    for _ in range(num_hyp_pars)
]

n_estimators = [
    XGBoost_helper.make_sample(samp_type="integer", min_lim=2, max_lim=256, scale="log")
    for _ in range(num_hyp_pars)
]

parameters_mat_rand = {
    "learning_rates": learning_rates,
    "min_child_weight": min_child_weight,
    "max_depth": max_depth,
    "n_estimators": n_estimators,
}

pickle.dump(
    parameters_mat_rand,
    open(
        "hyperparameters_file_random_num-%s_seed-%s.p" % (num_hyp_pars, gen_seed), "wb"
    ),
)
