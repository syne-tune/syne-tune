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

from sagemaker.estimator import Framework
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger(__name__)


class CustomFramework(Framework):

    __framework_name__ = "customframework"

    LATEST_VERSION = "0.1"

    def __init__(
        self,
        entry_point,
        image_uri: str,
        source_dir=None,
        hyperparameters=None,
        **kwargs
    ) -> None:
        super(CustomFramework, self).__init__(
            str(entry_point), source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
    ):
        # required to allow this object instantiation
        raise NotImplementedError()
