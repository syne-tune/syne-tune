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
from syne_tune.backend.sagemaker_backend.instance_info import select_instance_type, InstanceInfos


def test_instance_info():
    instance_infos = InstanceInfos()
    for instance in select_instance_type(max_gpu=0):
        assert instance_infos(instance).num_gpu == 0

    for instance in select_instance_type(min_gpu=1):
        assert instance_infos(instance).num_gpu >= 1

    for instance in select_instance_type(min_cost_per_hour=0.5, max_cost_per_hour=4.0):
        cost = instance_infos(instance).cost_per_hour
        assert 0.5 <= cost <= 4.0
