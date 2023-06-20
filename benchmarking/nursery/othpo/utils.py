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
import json
import numpy as np
import copy


def load_json_res(file_name):
    res = json.load(open(file_name, "r"))
    for key in ["data_sizes", "execution_times"]:
        res[key] = np.array(res[key])
    return res


def store_json_res(res, file_name):
    res_copy = copy.deepcopy(res)

    for key in ["data_sizes", "execution_times"]:
        res_copy[key] = res[key].tolist()

    for key in ["train_error_mat", "test_error_mat"]:
        res_copy[key] = [
            [[int(ii) for ii in ll] for ll in outer_ll] for outer_ll in res[key]
        ]
    json.dump(res_copy, open(file_name, "w"))
