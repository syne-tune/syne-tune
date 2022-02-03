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
from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace
from sagemaker.estimator import Framework

from syne_tune.backend.sagemaker_backend.custom_framework import \
    CustomFramework
from syne_tune.backend.sagemaker_backend.sagemaker_utils import \
    get_execution_role

logger = logging.getLogger(__name__)


def sagemaker_estimator_factory(
        entry_point: str, instance_type: str, framework: str = None,
        role: str = None, instance_count: int = 1,
        framework_version: str = None, py_version: str = None,
        dependencies: list = None, **kwargs) -> Framework:
    if role is None:
        role = get_execution_role()
    if py_version is None:
        py_version = 'py3'
    common_kwargs = dict(
        kwargs,
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
    )
    if dependencies is not None:
        common_kwargs['dependencies'] = dependencies
    if framework == 'PyTorch':
        sm_estimator = PyTorch(
            entry_point,
            framework_version=framework_version,
            py_version=py_version,
            **common_kwargs,
        )
    elif framework == 'HuggingFace':
        sm_estimator = HuggingFace(
            py_version,
            entry_point,
            **common_kwargs,
            transformers_version=framework_version,
        )
    else:
        if framework is not None:
            logger.info(
                f"framework = '{framework}' not supported, using "
                "CustomFramework")
        assert kwargs.get('image_uri') is not None, \
            "CustomFramework requires 'image_uri' to be specified"
        sm_estimator = CustomFramework(
            entry_point, **common_kwargs)
    return sm_estimator
