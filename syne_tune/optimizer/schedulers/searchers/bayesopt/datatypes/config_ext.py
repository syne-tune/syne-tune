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
from typing import Tuple
import copy

from syne_tune.config_space import randint
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration

RESOURCE_ATTR_PREFIX = "RESOURCE_ATTR_"


class ExtendedConfiguration:
    """
    This class facilitates handling extended configs, which consist of a normal
    config and a resource attribute.

    The config space hp_ranges is extended by an additional resource
    attribute. Note that this is not a hyperparameter we optimize over,
    but it is under the control of the scheduler.
    Its allowed range is [1, resource_attr_range[1]], which can be larger than
    [resource_attr_range[0], resource_attr_range[1]]. This is because extended
    configs with resource values outside of resource_attr_range may arise (for
    example, in the early stopping context, we may receive data from
    epoch < resource_attr_range[0]).

    """

    def __init__(
        self,
        hp_ranges: HyperparameterRanges,
        resource_attr_key: str,
        resource_attr_range: Tuple[int, int],
    ):
        assert resource_attr_range[0] >= 1
        assert resource_attr_range[1] >= resource_attr_range[0]
        self.hp_ranges = hp_ranges
        self.resource_attr_key = resource_attr_key
        self.resource_attr_range = resource_attr_range
        # Extended configuration space including resource attribute
        config_space_ext = copy.deepcopy(hp_ranges.config_space)
        self.resource_attr_name = RESOURCE_ATTR_PREFIX + resource_attr_key
        # Allowed range: [1, resource_attr_range[1]]
        assert self.resource_attr_name not in config_space_ext, (
            f"key = {self.resource_attr_name} is reserved, but appears in "
            + f"config_space = {list(config_space_ext.keys())}"
        )
        config_space_ext[self.resource_attr_name] = randint(
            lower=1, upper=resource_attr_range[1]
        )
        self.hp_ranges_ext = type(hp_ranges)(
            config_space_ext, name_last_pos=self.resource_attr_name
        )

    def get(self, config: Configuration, resource: int) -> Configuration:
        """
        Create extended config with resource added.

        :param config:
        :param resource:
        :return: Extended config
        """
        values = copy.copy(config)
        values[self.resource_attr_name] = resource
        return values

    def remove_resource(self, config_ext: Configuration) -> Configuration:
        """
        Strips away resource attribute and returns normal config. If
        `config_ext` is already normal, it is returned as is.

        :param config_ext: Extended config
        :return: config_ext without resource attribute
        """
        if self.resource_attr_name in config_ext:
            config = {
                k: v for k, v in config_ext.items() if k != self.resource_attr_name
            }
        else:
            config = config_ext
        return config

    def split(self, config_ext: Configuration) -> (Configuration, int):
        """
        Split extended config into normal config and resource value.

        :param config_ext: Extended config
        :return: (config, resource_value)
        """
        x_res = copy.copy(config_ext)
        resource_value = int(x_res[self.resource_attr_name])
        del x_res[self.resource_attr_name]
        return x_res, resource_value

    def get_resource(self, config_ext: Configuration) -> int:
        """
        :param config_ext: Extended config
        :return: Value of resource attribute
        """
        return int(config_ext[self.resource_attr_name])
