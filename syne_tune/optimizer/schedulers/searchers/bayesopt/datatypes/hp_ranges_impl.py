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
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import numpy as np

from syne_tune.search_space import Domain, is_log_space, FiniteRange
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Hyperparameter, Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.scaling \
    import Scaling, LinearScaling, LogScaling

__all__ = ['HyperparameterRangesImpl']


# This is code (slightly modified) from AmperCorePython. It is rather slow.
# Making use of Pandas would be faster.

# Epsilon margin to account for numerical errors
EPS = 1e-8


class HyperparameterRange(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        pass

    @abstractmethod
    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        pass

    def ndarray_size(self) -> int:
        return 1

    @abstractmethod
    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        pass


def scale_from_zero_one(
        value: float, lower_bound: float, upper_bound: float, scaling: Scaling,
        lower_internal: float, upper_internal: float):
    assert -EPS <= value <= 1.0 + EPS, value
    range = upper_internal - lower_internal
    hp = lower_bound
    if range > 0:
        internal_value = value * range + lower_internal
        hp = np.clip(
            scaling.from_internal(internal_value), lower_bound, upper_bound)
    return hp


class HyperparameterRangeContinuous(HyperparameterRange):
    def __init__(
            self, name: str, lower_bound: float, upper_bound: float,
            scaling: Scaling, active_lower_bound: float = None,
            active_upper_bound: float = None):
        """
        Real valued hyperparameter.
        If `active_lower_bound` and/or `active_upper_bound` are given, the
        feasible interval for values of new configs is reduced, but data can
        still contain configs with values in `[lower_bound, upper_bound]`, and
        internal encoding is done w.r.t. this original range.

        :param name: unique name of the hyperparameter.
        :param lower_bound: inclusive lower bound on all the values that
            parameter can take.
        :param upper_bound: inclusive upper bound on all the values that
            parameter can take.
        :param scaling: determines how the values of the parameter are enumerated internally.
            The parameter value is expressed as parameter = scaling(internal), where internal
            is internal representation of parameter, which is a real value, normally in range
            [0, 1]. To optimize the parameter, the internal is varied, and the parameter to be
            tested is calculated from such internal representation.
        :param active_lower_bound: See above
        :param active_upper_bound: See above
        """
        super().__init__(name)
        assert lower_bound <= upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scaling = scaling
        self.lower_internal = scaling.to_internal(lower_bound)
        self.upper_internal = scaling.to_internal(upper_bound)
        if active_lower_bound is None:
            active_lower_bound = lower_bound
        if active_upper_bound is None:
            active_upper_bound = upper_bound
        assert lower_bound <= active_upper_bound <= upper_bound
        assert lower_bound <= active_lower_bound <= upper_bound
        assert active_lower_bound <= active_upper_bound
        self._ndarray_bounds = [(self.to_ndarray(active_lower_bound)[0],
                                 self.to_ndarray(active_upper_bound)[0])]

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        assert self.lower_bound - EPS <= hp <= self.upper_bound + EPS, (hp, self)
        # convert everything to internal scaling, and then normalize between zero and one
        lower, upper = self.lower_internal, self.upper_internal
        if upper == lower:
            result = 0.0  # if the bounds are fixed for a dimension
        else:
            hp_internal = self.scaling.to_internal(hp)
            result = np.clip((hp_internal - lower) / (upper - lower), 0.0, 1.0)
        return np.array([result])

    def from_ndarray(self, ndarray: np.ndarray) -> Hyperparameter:
        return scale_from_zero_one(
            ndarray.item(), self.lower_bound, self.upper_bound, self.scaling,
            self.lower_internal, self.upper_internal)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__, repr(self.name),
            repr(self.scaling), repr(self.lower_bound), repr(self.upper_bound))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangeContinuous):
            return self.name == other.name \
                   and self.lower_bound == other.lower_bound \
                   and self.upper_bound == other.upper_bound \
                   and self.scaling == other.scaling
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._ndarray_bounds


class HyperparameterRangeInteger(HyperparameterRange):
    def __init__(
            self, name: str, lower_bound: int, upper_bound: int,
            scaling: Scaling, active_lower_bound: int = None,
            active_upper_bound: int = None):
        """
        Both bounds are INCLUDED in the valid values. Under the hood generates a continuous
        range from lower_bound - 0.5 to upper_bound + 0.5.
        See docs for continuous hyperparameter for more information.
        """
        super().__init__(name)
        assert lower_bound <= upper_bound
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.active_lower_bound = self.lower_bound if active_lower_bound is None \
            else int(active_lower_bound)
        self.active_upper_bound = self.upper_bound if active_upper_bound is None \
            else int(active_upper_bound)
        self._continuous_range = HyperparameterRangeContinuous(
            name, self.lower_bound - 0.5 + EPS, self.upper_bound + 0.5 - EPS,
            scaling, self.active_lower_bound - 0.5 + EPS,
            self.active_upper_bound + 0.5 - EPS)

    @property
    def scaling(self) -> Scaling:
        return self._continuous_range.scaling

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        return self._continuous_range.to_ndarray(float(hp))

    def _round_to_int(self, value: float) -> int:
        return np.clip(int(round(value)), self.lower_bound, self.upper_bound)

    def from_ndarray(self, ndarray: np.ndarray) -> Hyperparameter:
        continuous = self._continuous_range.from_ndarray(ndarray)
        return self._round_to_int(continuous)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__, repr(self.name),
            repr(self.scaling), repr(self.lower_bound), repr(self.upper_bound))

    def __eq__(self, other):
        if isinstance(other, HyperparameterRangeInteger):
            return self.name == other.name \
                   and self.lower_bound == other.lower_bound \
                   and self.upper_bound == other.upper_bound \
                   and self.scaling == other.scaling
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._continuous_range.get_ndarray_bounds()


class HyperparameterRangeFiniteRange(HyperparameterRange):
    def __init__(
            self, name: str, lower_bound: float, upper_bound: float,
            size: int, scaling: Scaling):
        """
        See :class:`FiniteRange` in `search_space`. Internally, we use an int
        with linear scaling.
        Note: Different to `HyperparameterRangeContinuous`, we require that
        `lower_bound < upper_bound` and `size >=2`.

        """
        super().__init__(name)
        assert lower_bound < upper_bound
        assert size >= 2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._scaling = scaling
        self._lower_internal = scaling.to_internal(lower_bound)
        self._upper_internal = scaling.to_internal(upper_bound)
        self._step_internal = \
            (self._upper_internal - self._lower_internal) / (size - 1)
        self._range_int = HyperparameterRangeInteger(
            name=name + '_INTERNAL', lower_bound=0, upper_bound=size - 1,
            scaling=LinearScaling())

    @property
    def scaling(self) -> Scaling:
        return self._scaling

    def _map_from_int(self, x: int) -> float:
        y = x * self._step_internal + self._lower_internal
        return np.clip(self._scaling.from_internal(y), self.lower_bound,
                       self.upper_bound)

    def _map_to_int(self, y: float) -> int:
        y_int = np.clip(self._scaling.to_internal(y), self._lower_internal,
                        self._upper_internal)
        return int(round((y_int - self._lower_internal) / self._step_internal))

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        return self._range_int.to_ndarray(self._map_to_int(hp))

    def from_ndarray(self, ndarray: np.ndarray) -> Hyperparameter:
        int_val = self._range_int.from_ndarray(ndarray)
        return self._map_from_int(int_val)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__, repr(self.name),
            repr(self.scaling), repr(self.lower_bound), repr(self.upper_bound))

    def __eq__(self, other):
        if isinstance(other, HyperparameterRangeFiniteRange):
            return self.name == other.name \
                   and self.lower_bound == other.lower_bound \
                   and self.upper_bound == other.upper_bound \
                   and self._scaling == other._scaling \
                   and self._range_int.upper_bound == other.   _range_int.upper_bound
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._range_int.get_ndarray_bounds()


class HyperparameterRangeCategorical(HyperparameterRange):
    def __init__(
            self, name: str, choices: Tuple[str, ...],
            active_choices: Tuple[str, ...] = None):
        """
        Can take on discrete set of values.
        :param name: name of dimension.
        :param choices: possible values of the hyperparameter.
        :param active_choices: If given, must be subset of `choices`.
        """
        super().__init__(name)
        self.choices = sorted([str(x) for x in choices])
        self.num_choices = len(self.choices)
        assert self.num_choices > 1
        if active_choices is None:
            self._ndarray_bounds = [(0.0, 1.0)] * self.num_choices
        else:
            _active_choices = set(active_choices)
            self._ndarray_bounds = [(0.0, 0.0)] * self.num_choices
            num = 0
            for pos, val in enumerate(self.choices):
                if val in _active_choices:
                    self._ndarray_bounds[pos] = (0.0, 1.0)
                    num += 1
            assert num == len(active_choices), \
                f"active_choices = {active_choices} must be a subset of " +\
                f"choices = {choices}"

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        hp = str(hp)
        assert hp in self.choices, "{} not in {}".format(hp, self)
        idx = self.choices.index(hp)
        result = np.zeros(shape=(self.num_choices,))
        result[idx] = 1.0
        return result

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        assert len(cand_ndarray) == self.num_choices, (cand_ndarray, self)
        return self.choices[int(np.argmax(cand_ndarray))]

    def ndarray_size(self) -> int:
        return self.num_choices

    def __repr__(self) -> str:
        return "{}({}, {})".format(
            self.__class__.__name__, repr(self.name), repr(self.choices)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperparameterRangeCategorical):
            return self.name == other.name \
                   and self.choices == other.choices
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._ndarray_bounds


class HyperparameterRangesImpl(HyperparameterRanges):
    """
    Basic implementation of :class:`HyperparameterRanges`. This is using code
    from AmperCorePython, which can be pretty slow.

    """
    def __init__(self, config_space: Dict, name_last_pos: str = None,
                 value_for_last_pos=None, active_config_space: Dict = None):
        super().__init__(config_space, name_last_pos, value_for_last_pos,
                         active_config_space)
        hp_ranges = []
        for name in self.internal_keys:
            hp_range = self.config_space[name]
            assert isinstance(hp_range, Domain)
            is_log = is_log_space(hp_range)
            tp = hp_range.value_type
            if tp == str:
                if name in self.active_config_space:
                    active_choices = tuple(
                        self.active_config_space[name].categories)
                else:
                    active_choices = None
                hp_ranges.append(HyperparameterRangeCategorical(
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
                        **kwargs, size=len(hp_range)))
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
            offset = self.ndarray_size() - enc_fixed.size
            for i, val in enumerate(enc_fixed):
                bounds[i + offset] = (val, val)
        return bounds

    def __repr__(self) -> str:
        return "{}{}".format(
            self.__class__.__name__, repr(self._hp_ranges)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangesImpl):
            return self._hp_ranges == other._hp_ranges
        return False
