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
from typing import Tuple, Dict, List, Optional
import numpy as np

from syne_tune.config_space import Domain, is_log_space, FiniteRange, Categorical
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.scaling \
    import LinearScaling, LogScaling
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl \
    import HyperparameterRangeContinuous, HyperparameterRangeInteger, \
    HyperparameterRangeCategoricalNonBinary, \
    HyperparameterRangeCategoricalBinary, HyperparameterRangeFiniteRange

__all__ = ['HyperparameterRangesImplNew']


class HyperparameterRangesImplNew(HyperparameterRanges):
    """
    Basic implementation of :class:`HyperparameterRanges`. 
    """
    def __init__(self, config_space: Dict, name_last_pos: str = None,
                 value_for_last_pos=None, active_config_space: Dict = None,
                 prefix_keys: Optional[List[str]] = None):
        super().__init__(config_space, name_last_pos, value_for_last_pos,
                         active_config_space, prefix_keys)
        hp_ranges = []
        for name in self.internal_keys:
            hp_range = self.config_space[name]
            assert isinstance(hp_range, Domain)
            is_log = is_log_space(hp_range)
            tp = hp_range.value_type
            if isinstance(hp_range, Categorical):
                if name in self.active_config_space:
                    active_choices = tuple(
                        self.active_config_space[name].categories)
                else:
                    active_choices = None
                if len(hp_range.categories) == 2:
                    _cls = HyperparameterRangeCategoricalBinary
                else:
                    _cls = HyperparameterRangeCategoricalNonBinary
                hp_ranges.append(_cls(
                    name, choices=tuple(hp_range.categories),
                    active_choices=active_choices))
            else:
                scaling = LogScaling() if is_log else LinearScaling()
                kwargs = {
                    'name': name,
                    'lower_bound': hp_range.lower,
                    'upper_bound': hp_range.upper,
                    'scaling': scaling}
                if isinstance(hp_range, FiniteRange):
                    assert name not in self.active_config_space, \
                        f"Parameter '{name}' of type FiniteRange cannot be used in active_config_space"
                    hp_ranges.append(HyperparameterRangeFiniteRange(
                        **kwargs, size=len(hp_range), cast_int=hp_range.cast_int))
                else:
                    # Note: If `hp_range` is logarithmic, it has a base.
                    # Since both the loguniform distribution and the internal
                    # encoding are independent of this base, we can just ignore
                    # it here (we use natural logarithms internally).
                    if name in self.active_config_space:
                        active_hp_range = self.active_config_space[name]
                        kwargs.update({
                            'active_lower_bound': active_hp_range.lower,
                            'active_upper_bound': active_hp_range.upper})
                    if tp == float:
                        hp_ranges.append(HyperparameterRangeContinuous(**kwargs))
                    else:
                        hp_ranges.append(HyperparameterRangeInteger(**kwargs))
        self._hp_ranges = hp_ranges
        self._ndarray_size = sum(d.ndarray_size() for d in hp_ranges)

    @property
    def ndarray_size(self) -> int:
        return self._ndarray_size

    def to_ndarray(self, config: Configuration) -> np.ndarray:
        config_tpl = self.config_to_tuple(config)
        pieces = [hp_range.to_ndarray(hp)
                  for hp_range, hp in zip(self._hp_ranges, config_tpl)]
        return np.hstack(pieces)

    def from_ndarray(self, enc_config: np.ndarray) -> Configuration:
        """
        Converts a config from internal ndarray representation (fed to the GP)
        into an external config.

        For numerical HPs it assumes values scaled between 0.0 and 1.0, for
        categorical HPs it assumes one scalar per category, which will convert
        to the category with the highest value.
        """
        enc_config = enc_config.reshape((-1, 1))
        assert enc_config.size == self._ndarray_size, \
            (enc_config.size, self._ndarray_size)
        hps = []
        start = 0
        for hp_range in self._hp_ranges:
            end = start + hp_range.ndarray_size()
            enc_attr = enc_config[start:end]
            hps.append(hp_range.from_ndarray(enc_attr))
            start = end
        return self.tuple_to_config(tuple(hps))

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        bounds = [x for hp_range in self._hp_ranges
                  for x in hp_range.get_ndarray_bounds()]
        if self.is_attribute_fixed():
            hp_range = self._hp_ranges[-1]
            assert hp_range.name == self.name_last_pos
            enc_fixed = hp_range.to_ndarray(
                self.value_for_last_pos).reshape((-1,))
            offset = self.ndarray_size - enc_fixed.size
            for i, val in enumerate(enc_fixed):
                bounds[i + offset] = (val, val)
        return bounds

    def __repr__(self) -> str:
        return "{}{}".format(
            self.__class__.__name__, repr(self._hp_ranges)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangesImplNew):
            return self._hp_ranges == other._hp_ranges
        return False
