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
import numpy as np

from syne_tune.optimizer.schedulers.multiobjective.utils import hypervolume


def test_hypervolume():
    ref_point = np.array([2, 2])
    points = np.array([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
    hv = hypervolume(results_array=points, reference_points=ref_point)
    assert hv(points) == 3.25
