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
"""
An entry point that loads a serialized function from `PythonBackend` and executes it with the provided hyperparameter.
The md5 hash of the file is first checked before executing the deserialized function.
"""
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import dill

from syne_tune.backend.python_backend.python_backend import file_md5
from syne_tune.config_space import add_to_argparse, from_dict

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(f"--tune_function_root", type=str)
    parser.add_argument(f"--tune_function_hash", type=str)
    args, _ = parser.parse_known_args()

    # first parse args to get where the function and config space were saved and
    # check the md5 of the serialized function is the same
    # then parse args again with parameters defined in the config space
    assert args.tune_function_root
    assert args.tune_function_hash
    root = Path(args.tune_function_root)
    assert (
        file_md5(root / "tune_function.dill") == args.tune_function_hash
    ), "The hash of the tuned function should match the hash obtained when serializing in Syne Tune."
    with open(root / "tune_function.dill", "rb") as file:
        tuned_function = dill.load(file)

    with open(root / "configspace.json", "r") as file:
        config_space = json.load(file)
        config_space = {
            k: from_dict(v) if isinstance(v, Dict) else v
            for k, v in config_space.items()
        }

    add_to_argparse(parser, config_space)

    args, _ = parser.parse_known_args()

    hps = {k: v for k, v in args.__dict__.items() if k in config_space}
    tuned_function(**hps)
