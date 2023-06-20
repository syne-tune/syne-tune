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
import pickle
import pandas as pd
import numpy as np

from experiment_master_file import experiments_meta_dict, get_task_values, get_exp_title
from backend_definitions_dict import BACKEND_DEFS

for experiment in ["SimOpt", "XGBoost", "YAHPO_auc_svm_1220"]:

    preprocessed_results = pickle.load(
        open("plotting/preprocessed_results_%s.p" % experiment, "rb")
    )

    experiments_meta_data = experiments_meta_dict[experiment]
    print(get_exp_title(experiments_meta_data))

    task_values = get_task_values(experiments_meta_data)

    opt_mode = BACKEND_DEFS[experiments_meta_data["backend"]][1]

    to_make_df = {}
    it = 0
    for method in ["WarmBO", "Quantiles"]:
        to_make_df["%s mean" % method] = [
            preprocessed_results["mean_performance"][method][task_val][it]
            for task_val in task_values
        ]
        to_make_df["%s mean std" % method] = [
            preprocessed_results["std_mean_performance"][method][task_val][it]
            for task_val in task_values
        ]

    df = pd.DataFrame(to_make_df)

    if opt_mode == "max":
        df["perc. improv"] = 100 * (df["WarmBO mean"] / df["Quantiles mean"] - 1)
    else:
        df["perc. improv"] = 100 * (1 - df["WarmBO mean"] / df["Quantiles mean"])
    df["perc. reduction std"] = 100 * (
        1 - df["WarmBO mean std"] / df["Quantiles mean std"]
    )

    transfer_df = df[1:]

    mean_imp = transfer_df["perc. improv"].mean()
    sem_imp = transfer_df["perc. improv"].sem(ddof=1)
    mean_print = np.round(mean_imp, 1)
    low_print = np.round(mean_imp - 2 * sem_imp, 1)
    high_print = np.round(mean_imp + 2 * sem_imp, 1)

    mean_imp_std = transfer_df["perc. reduction std"].mean()
    sem_imp_std = transfer_df["perc. reduction std"].sem(ddof=1)
    mean_print_std = np.round(mean_imp_std, 1)
    low_print_std = np.round(mean_imp_std - 2 * sem_imp_std, 1)
    high_print_std = np.round(mean_imp_std + 2 * sem_imp_std, 1)

    print("Perc. improv: %s (%s -- %s)" % (mean_print, low_print, high_print))
    print(
        "Perc. reduction std: %s (%s -- %s)"
        % (mean_print_std, low_print_std, high_print_std)
    )
