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
from typing import Tuple, List, Iterable, Dict, Optional
import numpy as np
from numpy.random import RandomState

from syne_tune.config_space import (
    non_constant_hyperparameter_keys,
    is_log_space,
    config_to_match_string,
    is_reverse_log_space,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Hyperparameter,
    Configuration,
)

__all__ = ["HyperparameterRanges"]


def _filter_constant_hyperparameters(config_space: Dict) -> Dict:
    nonconst_keys = set(non_constant_hyperparameter_keys(config_space))
    return {k: v for k, v in config_space.items() if k in nonconst_keys}


class HyperparameterRanges:
    def __init__(
        self,
        config_space: Dict,
        name_last_pos: Optional[str] = None,
        value_for_last_pos=None,
        active_config_space: Optional[Dict] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        """
        If name_last_pos is given, the hyperparameter of that name is assigned
        the final position in the vector returned by `to_ndarray`. This can be
        used to single out the (time) resource for a GP model, where that
        component has to come last.

        If in this case (name_last_pos given), value_for_last_pos is also given,
        some methods are modified:
        - `random_config` samples a config as normal, but then overwrites the
          name_last_pos component by value_for_last_pos
        - `get_ndarray_bounds` works as normal, but returns bound (a, a) for
          name_last_pos component, where a is the internal value corresponding
          to value_for_last_pos
        The use case is HPO with a resource attribute. This attribute should be
        fixed when optimizing the acquisition function, but can take different
        values in the evaluation data (coming from all previous searches).

        If `active_config_space` is given, it contains a subset of non-constant
        hyperparameters in `config_space`, and the range of each entry is a
        subset of the range of the corresponding `config_space` entry. These
        active ranges affect the choice of new configs (by sampling). While the
        internal encoding is based on original ranges, search is restricted to
        active ranges (e.g., optimization of surrogate model).

        :param config_space: Configuration space. Constant hyperparameters are
            filtered out here
        :param name_last_pos: See above
        :param value_for_last_pos: See above
        :param active_config_space: See above
        :param prefix_keys: If given, these keys into `config_space` come first
            in the internal ordering, which determines the internal
            encoding
        """
        self.config_space = _filter_constant_hyperparameters(config_space)
        self.name_last_pos = name_last_pos
        self.value_for_last_pos = value_for_last_pos
        self._set_internal_keys(prefix_keys)
        self._set_active_config_space(active_config_space)

    def _set_internal_keys(self, prefix_keys: Optional[List[str]]):
        keys = sorted(self.config_space.keys())
        if prefix_keys is not None:
            pk_set = set(prefix_keys)
            assert pk_set.issubset(
                set(keys)
            ), f"prefix_keys = {prefix_keys} is not a subset of {keys}"
            keys = prefix_keys + [key for key in keys if key not in pk_set]
        if self.name_last_pos is not None:
            assert self.name_last_pos in keys, (
                f"name_last_pos = '{self.name_last_pos}' not among "
                + f"hyperparameter names [{keys}]"
            )
            pos = keys.index(self.name_last_pos)
            keys = keys[:pos] + keys[(pos + 1) :] + [self.name_last_pos]
        self._internal_keys = keys

    def _set_active_config_space(self, active_config_space: Dict):
        if active_config_space is None:
            self.active_config_space = dict()
            self._config_space_for_sampling = self.config_space
        else:
            self._assert_sub_config_space(active_config_space)
            self.active_config_space = active_config_space
            self._config_space_for_sampling = dict(
                self.config_space, **active_config_space
            )

    def _assert_sub_config_space(self, active_config_space: Dict):
        for k, v in active_config_space.items():
            assert (
                k in self.config_space
            ), f"active_config_space[{k}] not in config_space"
            same_value_type = v.value_type == self.config_space[k].value_type
            same_log_type = is_log_space(v) == is_log_space(
                self.config_space[k]
            ) and is_reverse_log_space(v) == is_reverse_log_space(self.config_space[k])
            same_domain_type = isinstance(v, type(self.config_space[k]))
            assert (
                k in self.config_space
                and same_value_type
                and same_log_type
                and same_domain_type
            ), f"active_config_space[{k}] has different type"

    @property
    def internal_keys(self) -> List[str]:
        return self._internal_keys

    @property
    def config_space_for_sampling(self) -> Dict:
        return self._config_space_for_sampling

    def to_ndarray(self, config: Configuration) -> np.ndarray:
        """
        Categorical values are one-hot encoded.

        :param config: Config to encode
        :return: Encoded HP vector
        """
        raise NotImplementedError()

    def to_ndarray_matrix(self, configs: Iterable[Configuration]) -> np.ndarray:
        return np.vstack([self.to_ndarray(config) for config in configs])

    @property
    def ndarray_size(self) -> int:
        """
        :return: Dimensionality of encoded HP vector returned by `to_ndarray`
        """
        raise NotImplementedError()

    def from_ndarray(self, enc_config: np.ndarray) -> Configuration:
        """
        Converts a config from internal ndarray representation (fed to the
        GP) to its external (dict) representation. This typically involves
        rounding.
        """
        raise NotImplementedError()

    @property
    def encoded_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        :return: Ranges of HPs in the encoded ndarray representation
        """
        raise NotImplementedError()

    def is_attribute_fixed(self):
        return (self.name_last_pos is not None) and (
            self.value_for_last_pos is not None
        )

    def _fix_attribute_value(self, name):
        return self.is_attribute_fixed() and name == self.name_last_pos

    def _transform_config(self, config: Configuration):
        if self.is_attribute_fixed():
            config[self.name_last_pos] = self.value_for_last_pos
        return config

    def _random_config(self, random_state: RandomState) -> Configuration:
        return {
            k: v.sample(random_state=random_state)
            for k, v in self._config_space_for_sampling.items()
        }

    def random_config(self, random_state: RandomState) -> Configuration:
        return self._transform_config(self._random_config(random_state))

    def _random_configs(
        self, random_state: RandomState, num_configs: int
    ) -> List[Configuration]:
        return [self._random_config(random_state) for _ in range(num_configs)]

    def random_configs(self, random_state, num_configs: int) -> List[Configuration]:
        return [
            self._transform_config(config)
            for config in self._random_configs(random_state, num_configs)
        ]

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        """
        Returns (lower, upper) bounds for each dimension in ndarray vector
        representation.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.config_space)

    def filter_for_last_pos_value(
        self, configs: List[Configuration]
    ) -> List[Configuration]:
        """
        If is_attribute_fixed, `configs` is filtered by removing
        entries whose name_last_pos attribute value is different from
        value_for_last_pos. Otherwise, it is returned unchanged.

        """
        if self.is_attribute_fixed():
            configs = [
                config
                for config in configs
                if config[self.name_last_pos] == self.value_for_last_pos
            ]
        return configs

    def config_to_tuple(
        self, config: Configuration, keys=None, skip_last: bool = False
    ) -> Tuple[Hyperparameter, ...]:
        """
        :param config: Configuration
        :param keys: Overrides `_internal_keys`
        :param skip_last: If True and `name_last_pos` is used, the
            corresponding attribute is skipped, so that config and
            tuple are non-extended
        :return: Tuple representation
        """
        if keys is None:
            keys = self.internal_keys
            if skip_last and self.name_last_pos is not None:
                keys = keys[:-1]  # Skip last pos
        return tuple(config[k] for k in keys)

    def tuple_to_config(
        self, config_tpl: Tuple[Hyperparameter, ...], keys=None, skip_last: bool = False
    ) -> Configuration:
        if keys is None:
            keys = self.internal_keys
            if skip_last and self.name_last_pos is not None:
                keys = keys[:-1]  # Skip last pos
        return dict(zip(keys, config_tpl))

    def config_to_match_string(
        self, config: Configuration, keys=None, skip_last: bool = False
    ) -> str:
        """
        Maps configuration to  match string, used to compare for approximate
        equality. Two configurations are considered to be different if their
        match strings are not the same.

        :param config: Configuration
        :param keys: Overrides `_internal_keys`
        :param skip_last: If True and `name_last_pos` is used, the
            corresponding attribute is skipped, so that config and
            tuple are non-extended
        :return: Match string
        """
        if keys is None:
            keys = self.internal_keys
            if skip_last and self.name_last_pos is not None:
                keys = keys[:-1]  # Skip last pos
        return config_to_match_string(config, self.config_space, keys)
