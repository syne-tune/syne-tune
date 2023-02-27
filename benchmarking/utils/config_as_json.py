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
from typing import Dict, Any
import argparse
import json

from syne_tune.constants import ST_CONFIG_JSON_FNAME_ARG


def add_config_json_to_argparse(parser: argparse.ArgumentParser):
    """
    To be called for the argument parser in the endpoint script.

    :param parser: Parser to add extra arguments to
    """
    parser.add_argument(f"--{ST_CONFIG_JSON_FNAME_ARG}", type=str)


def load_config_json(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads configuration from JSON file and returns the union with ``args``.

    :param args: Arguments returned by ``ArgumentParser``, as dictionary
    :return: Combined configuration dictionary
    """
    with open(args[ST_CONFIG_JSON_FNAME_ARG], "r") as f:
        config = json.load(f)
    config.update(args)
    return config
