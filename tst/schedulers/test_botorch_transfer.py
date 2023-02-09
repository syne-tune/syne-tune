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
import pytest
import pandas as pd
import numpy as np

from syne_tune.optimizer.schedulers.searchers.botorch.botorch_transfer_searcher import (
    BoTorchTransfer,
)

import syne_tune.config_space as sp

from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)

config_space = {
    "int_var": sp.randint(-2, 3),
    "con_var": sp.uniform(-1, 5),
    "cat_var": sp.choice(["A", "B", "C"]),
    "ord_var": sp.ordinal([1, 4, 10], kind="equal"),
}

samp1 = {"int_var": -2, "con_var": -0.9, "cat_var": "A", "ord_var": 1}

samp2 = {"int_var": -1, "con_var": -0.4, "cat_var": "C", "ord_var": 10}

samp3 = {"int_var": 3, "con_var": 3.4, "cat_var": "B", "ord_var": 4}

samp4 = {"int_var": 1, "con_var": 0.0, "cat_var": "C", "ord_var": 10}

samp5 = {"int_var": 0, "con_var": 3.1, "cat_var": "A", "ord_var": 4}

df_task_1 = pd.DataFrame([samp3, samp4, samp5])
evals_task_1 = np.array([0.3, 0.6, 0.9], ndmin=4).T

df_task_2 = pd.DataFrame([samp1, samp2])
evals_task_2 = np.ones((2, 1, 1, 1)) * 3

scheduler = BoTorchTransfer(
    config_space=config_space,
    metric="WhoKnows",
    new_task_id="3",
    transfer_learning_evaluations={
        "2": TransferLearningTaskEvaluations(
            config_space, df_task_2, ["WhoKnows"], evals_task_2
        ),
        "1": TransferLearningTaskEvaluations(
            config_space, df_task_1, ["WhoKnows"], evals_task_1
        ),
    },
)

old_confs = scheduler.searcher._configs_with_results()


def same_up_to_task(samp, samp_w_task):
    for key in samp:
        if samp[key] != samp_w_task[key]:
            return False
    return True


def check_valid_encoding(encoding):
    return ((encoding >= 0) & (encoding <= 1) & ~pd.isna(encoding)).all()


def test_transfer_samples_added():
    for samp in [samp1, samp2, samp3, samp4, samp5]:
        assert np.any(
            [same_up_to_task(samp, conf) for conf in old_confs]
        ), "Check that all expected former samples appear"


def test_valid_encoding_different_tasks():
    for task_val in ["1", "2", "3"]:
        encoding = scheduler.searcher._config_to_feature_matrix(
            [samp1], task_val=task_val
        )
        assert check_valid_encoding(encoding), "Check for infs and nans in encoding"


def test_valid_encoding_X_array():
    for encoding in np.array(scheduler.searcher._config_to_feature_matrix(old_confs)):
        assert check_valid_encoding(encoding), "Check for infs and nans in encoding"
