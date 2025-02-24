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
