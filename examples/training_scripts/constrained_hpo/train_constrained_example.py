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
import logging
import numpy as np

from syne_tune import Reporter
from argparse import ArgumentParser


report = Reporter()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    parser.add_argument('--constraint_offset', type=float)

    args, _ = parser.parse_known_args()

    x1 = args.x1
    x2 = args.x2
    constraint_offset = args.constraint_offset
    r = 6
    objective_value = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 +
                       (5 / np.pi) * x1 - r) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    constraint_value = x1 * 2.0 - constraint_offset  # feasible iff x1 <= 0.5 * constraint_offset
    report(objective=-objective_value, my_constraint_metric=constraint_value)
