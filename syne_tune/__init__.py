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

try:
    __all__ = ['config_space.py', 'StoppingCriterion', 'Tuner', 'Reporter']
    from pathlib import Path
    from syne_tune.stopping_criterion import StoppingCriterion
    from syne_tune.report import Reporter
    from syne_tune.tuner import Tuner
except ImportError:
    pass


def read_version():
    with open(Path(__file__).parent / "version.py", "r") as f:
        return f.readline().replace("\"", "")


__version__ = read_version()
