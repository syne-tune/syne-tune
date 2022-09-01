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
from pathlib import Path

try:
    # The reason for conditional imports is that `read_version` is called
    # by `setup.py` before any dependencies are installed
    from syne_tune.stopping_criterion import StoppingCriterion  # noqa: F401
    from syne_tune.report import Reporter  # noqa: F401
    from syne_tune.tuner import Tuner  # noqa: F401

    __all__ = ["StoppingCriterion", "Tuner", "Reporter"]
except ImportError:
    __all__ = []


def read_version():
    with open(Path(__file__).parent / "version", "r") as f:
        return f.readline().strip().replace('"', "")


__version__ = read_version()
