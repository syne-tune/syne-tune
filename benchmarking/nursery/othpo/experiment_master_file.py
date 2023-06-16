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
from blackbox_helper import get_configs


def get_exp_title(meta_dict):
    if meta_dict["backend"] == "XGBoost":
        return meta_dict["backend"]
    elif meta_dict["backend"] == "SimOpt":
        return "NewsVendor"
    else:
        scenario = meta_dict["yahpo_scenario"].split("_")[1]
        if scenario in ["svm", "aknn"]:
            scenario = scenario.upper()
        return "%s %s" % (scenario, meta_dict["yahpo_dataset"])


get_task_values = lambda experiments_meta_data: get_configs(
    backend=experiments_meta_data["backend"],
    xgboost_res_file=experiments_meta_data["xgboost_res_file"],
    simopt_backend_file=experiments_meta_data["simopt_backend_file"],
    yahpo_dataset=experiments_meta_data["yahpo_dataset"],
    yahpo_scenario=experiments_meta_data["yahpo_scenario"],
)[0]

experiments_meta_dict = {
    "SimOpt": {
        "backend": "SimOpt",
        "simopt_backend_file": "simopt/SimOptNewsPrice.py",
        "xgboost_res_file": None,
        "yahpo_scenario": None,
        "yahpo_dataset": None,
        "metric": None,
        "files": [
            "collect_results_2023-02-21-17-31-18.p",  # BoundingBox
            "collect_results_2023-02-22-13-59-53.p",  # ZeroShot
            "collect_results_2023-03-06-14-30-16.p",  # Quantiles
            "collect_results_2023-02-21-17-31-26.p",  # BayesOpt
            "collect_results_2023-02-21-17-31-32.p",  # WarmBo
            "collect_results_2023-02-21-18-45-55.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-49-18.p",  # BoTorchTransfer
            "collect_results_2023-03-08-23-49-23.p",  # RandomSearch
            "collect_results_2023-03-19-15-34-22.p",  # PrevBO
            "collect_results_2023-03-19-15-34-26.p",  # PrevNoBO
        ],
    },
    "XGBoost": {
        "backend": "XGBoost",
        "simopt_backend_file": None,
        "xgboost_res_file": "xgboost_experiment_results/random-mnist/aggregated_experiments.json",
        "yahpo_scenario": None,
        "yahpo_dataset": None,
        "metric": None,
        "files": [
            "collect_results_2023-02-22-15-56-42.p",  # BoundingBox
            "collect_results_2023-02-22-14-00-14.p",  # ZeroShot
            "collect_results_2023-03-06-14-30-23.p",  # Quantiles
            "collect_results_2023-02-21-17-31-46.p",  # BayesOpt
            "collect_results_2023-02-21-17-31-53.p",  # WarmBo
            "collect_results_2023-02-21-18-46-00.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-49-30.p",  # BoTorchTran
            "collect_results_2023-03-08-23-49-36.p",  # RandomSearch
            "collect_results_2023-03-19-15-34-31.p",  # PrevBO
            "collect_results_2023-03-19-15-34-35.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_svm_1220": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_svm",
        "yahpo_dataset": "1220",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-01-14-18-29.p",  # BoundingBox
            "collect_results_2023-03-01-14-18-36.p",  # ZeroShot
            "collect_results_2023-03-06-14-10-41.p",  # Quantiles
            "collect_results_2023-03-01-14-18-50.p",  # BayesOpt
            "collect_results_2023-03-01-14-18-54.p",  # WarmBo
            "collect_results_2023-03-01-14-18-58.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-49-43.p",  # BoTorchTran
            "collect_results_2023-03-08-23-49-50.p",  # RandomSearch
            "collect_results_2023-03-19-15-34-40.p",  # PrevBO
            "collect_results_2023-03-19-15-34-46.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_svm_458": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_svm",
        "yahpo_dataset": "458",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-03-16-30-17.p",  # WarmBo
            "collect_results_2023-03-03-16-30-38.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-52-41.p",  # RandomSearch
            "collect_results_2023-03-19-16-54-24.p",  # PrevBO
            "collect_results_2023-03-19-16-54-30.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_aknn_4538": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_aknn",
        "yahpo_dataset": "4538",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-01-21-13-01.p",  # WarmBo
            "collect_results_2023-03-01-21-13-12.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-52-48.p",  # RandomSearch
            "collect_results_2023-03-19-16-54-34.p",  # PrevBO
            "collect_results_2023-03-19-16-54-41.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_aknn_41138": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_aknn",
        "yahpo_dataset": "41138",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-03-16-30-25.p",  # WarmBo
            "collect_results_2023-03-03-16-30-45.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-52-55.p",  # RandomSearch
            "collect_results_2023-03-19-16-54-45.p",  # PrevBO
            "collect_results_2023-03-19-16-54-52.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_ranger_4154": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_ranger",
        "yahpo_dataset": "4154",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-01-21-13-08.p",  # WarmBo
            "collect_results_2023-03-01-21-13-19.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-53-01.p",  # RandomSearch
            "collect_results_2023-03-19-16-54-57.p",  # PrevBO
            "collect_results_2023-03-19-16-55-03.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_ranger_40978": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_ranger",
        "yahpo_dataset": "40978",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-03-16-30-32.p",  # WarmBo
            "collect_results_2023-03-03-16-30-52.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-53-08.p",  # RandomSearch
            "collect_results_2023-03-19-16-55-09.p",  # PrevBO
            "collect_results_2023-03-19-16-55-14.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_glmnet_375": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_glmnet",
        "yahpo_dataset": "375",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-03-16-39-07.p",  # WarmBo
            "collect_results_2023-03-03-16-39-21.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-53-12.p",  # RandomSearch
            "collect_results_2023-03-19-16-55-19.p",  # PrevBO
            "collect_results_2023-03-19-16-55-23.p",  # PrevNoBO
        ],
    },
    "YAHPO_auc_glmnet_40981": {
        "backend": "YAHPO",
        "simopt_backend_file": None,
        "xgboost_res_file": None,
        "yahpo_scenario": "rbv2_glmnet",
        "yahpo_dataset": "40981",
        "metric": "auc",
        "files": [
            "collect_results_2023-03-03-16-39-15.p",  # WarmBo
            "collect_results_2023-03-03-16-39-28.p",  # WarmBoShuff
            "collect_results_2023-03-08-23-53-20.p",  # RandomSearch
            "collect_results_2023-03-19-16-55-28.p",  # PrevBO
            "collect_results_2023-03-19-16-55-34.p",  # PrevNoBO
        ],
    },
}
