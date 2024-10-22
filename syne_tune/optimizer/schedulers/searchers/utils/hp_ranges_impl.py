from typing import Tuple, Dict, List, Any, Optional, Union
import numpy as np

from syne_tune.config_space import (
    Domain,
    FiniteRange,
    Categorical,
    Ordinal,
    OrdinalNearestNeighbor,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Hyperparameter,
    Configuration,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.scaling import (
    Scaling,
    LinearScaling,
    get_scaling,
)


# Epsilon margin to account for numerical errors
EPS = 1e-8


class HyperparameterRange:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        raise NotImplementedError

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        raise NotImplementedError

    def ndarray_size(self) -> int:
        return 1

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        raise NotImplementedError


def scale_from_zero_one(
    value: float,
    lower_bound: float,
    upper_bound: float,
    scaling: Scaling,
    lower_internal: float,
    upper_internal: float,
):
    assert -EPS <= value <= 1.0 + EPS, value
    size = upper_internal - lower_internal
    hp = lower_bound
    if size > 0:
        internal_value = value * size + lower_internal
        hp = np.clip(scaling.from_internal(internal_value), lower_bound, upper_bound)
    return hp


class HyperparameterRangeContinuous(HyperparameterRange):
    """
    Real valued hyperparameter.
    If ``active_lower_bound`` and/or ``active_upper_bound`` are given, the
    feasible interval for values of new configs is reduced, but data can
    still contain configs with values in ``[lower_bound, upper_bound]``, and
    internal encoding is done w.r.t. this original range.

    :param name: Name of hyperparameter
    :param lower_bound: Lower bound (included)
    :param upper_bound: Upper bound (included)
    :param scaling: Determines internal representation, whereby
        ``parameter = scaling(internal)``.
    :param active_lower_bound: See above
    :param active_upper_bound: See above
    """

    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        scaling: Scaling,
        active_lower_bound: float = None,
        active_upper_bound: float = None,
    ):
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
        self._ndarray_bounds = [
            (
                self.to_ndarray(active_lower_bound)[0],
                self.to_ndarray(active_upper_bound)[0],
            )
        ]

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
            ndarray.item(),
            self.lower_bound,
            self.upper_bound,
            self.scaling,
            self.lower_internal,
            self.upper_internal,
        )

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.scaling),
            repr(self.lower_bound),
            repr(self.upper_bound),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangeContinuous):
            return (
                self.name == other.name
                and np.allclose([self.lower_bound], [other.lower_bound])
                and np.allclose([self.upper_bound], [other.upper_bound])
                and self.scaling == other.scaling
            )
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._ndarray_bounds


class HyperparameterRangeInteger(HyperparameterRange):
    """
    Integer valued hyperparameter.
    Both bounds are *included* in the valid values. Under the hood generates
    a continuous range from ``lower_bound - 0.5`` to ``upper_bound + 0.5``.
    See docs for continuous hyperparameter for more information.

    :param name: Name of hyperparameter
    :param lower_bound: Lower bound (integer, included)
    :param upper_bound: Upper bound (integer, included)
    :param scaling: Determines internal representation, whereby
        ``parameter = scaling(internal)``.
    :param active_lower_bound: See above
    :param active_upper_bound: See above
    """

    def __init__(
        self,
        name: str,
        lower_bound: int,
        upper_bound: int,
        scaling: Scaling,
        active_lower_bound: int = None,
        active_upper_bound: int = None,
    ):
        super().__init__(name)
        assert lower_bound <= upper_bound
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.active_lower_bound = (
            self.lower_bound if active_lower_bound is None else int(active_lower_bound)
        )
        self.active_upper_bound = (
            self.upper_bound if active_upper_bound is None else int(active_upper_bound)
        )
        self._continuous_range = HyperparameterRangeContinuous(
            name,
            self.lower_bound - 0.5 + EPS,
            self.upper_bound + 0.5 - EPS,
            scaling,
            self.active_lower_bound - 0.5 + EPS,
            self.active_upper_bound + 0.5 - EPS,
        )

    @property
    def scaling(self) -> Scaling:
        return self._continuous_range.scaling

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        return self._continuous_range.to_ndarray(float(hp))

    def _round_to_int(self, value: float) -> int:
        return int(np.clip(round(value), self.lower_bound, self.upper_bound))

    def from_ndarray(self, ndarray: np.ndarray) -> Hyperparameter:
        continuous = self._continuous_range.from_ndarray(ndarray)
        return self._round_to_int(continuous)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.scaling),
            repr(self.lower_bound),
            repr(self.upper_bound),
        )

    def __eq__(self, other):
        if isinstance(other, HyperparameterRangeInteger):
            return (
                self.name == other.name
                and self.lower_bound == other.lower_bound
                and self.upper_bound == other.upper_bound
                and self.scaling == other.scaling
            )
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._continuous_range.get_ndarray_bounds()


class HyperparameterRangeFiniteRange(HyperparameterRange):
    """
    Finite range numerical hyperparameter, see
    :class:`~syne_tune.config_space.FiniteRange`. Internally, we use an ``int``
    with linear scaling.

    Note: Different to :class:`HyperparameterRangeContinuous`, we require that
    ``lower_bound < upper_bound`` and ``size >=2``.

    :param name: Name of hyperparameter
    :param lower_bound: Lower bound (included)
    :param upper_bound: Upper bound (included)
    :param size: Number of values in range
    :param scaling: Determines internal representation, whereby
        ``parameter = scaling(internal)``.
    :param cast_int: If True, values are cast to ``int``
    """

    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        size: int,
        scaling: Scaling,
        cast_int: bool = False,
    ):
        super().__init__(name)
        assert lower_bound <= upper_bound
        assert size >= 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.cast_int = cast_int
        self._scaling = scaling
        self._lower_internal = scaling.to_internal(lower_bound)
        self._upper_internal = scaling.to_internal(upper_bound)
        self._step_internal = (
            (self._upper_internal - self._lower_internal) / (size - 1)
            if size > 1
            else 0
        )
        self._range_int = HyperparameterRangeInteger(
            name=name + "_INTERNAL",
            lower_bound=0,
            upper_bound=size - 1,
            scaling=LinearScaling(),
        )

    @property
    def scaling(self) -> Scaling:
        return self._scaling

    def _map_from_int(self, x: int) -> Union[float, int]:
        y = x * self._step_internal + self._lower_internal
        y = np.clip(self._scaling.from_internal(y), self.lower_bound, self.upper_bound)
        if not self.cast_int:
            return float(y)
        else:
            return int(np.round(y))

    def _map_to_int(self, y: Union[float, int]) -> int:
        if self._step_internal == 0:
            return 0
        else:
            y_int = np.clip(
                self._scaling.to_internal(y), self._lower_internal, self._upper_internal
            )
            return int(round((y_int - self._lower_internal) / self._step_internal))

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        return self._range_int.to_ndarray(self._map_to_int(hp))

    def from_ndarray(self, ndarray: np.ndarray) -> Hyperparameter:
        int_val = self._range_int.from_ndarray(ndarray)
        return self._map_from_int(int_val)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {}, {})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.scaling),
            repr(self.lower_bound),
            repr(self.upper_bound),
            repr(self.cast_int),
        )

    def __eq__(self, other):
        if isinstance(other, HyperparameterRangeFiniteRange):
            return (
                self.name == other.name
                and np.allclose([self.lower_bound], [other.lower_bound])
                and np.allclose([self.upper_bound], [other.upper_bound])
                and self._scaling == other._scaling
                and self.cast_int == other.cast_int
                and self._range_int.upper_bound == other._range_int.upper_bound
            )
        return False

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._range_int.get_ndarray_bounds()


class HyperparameterRangeCategorical(HyperparameterRange):
    """
    Base class for categorical hyperparameter.

    :param name: Name of hyperparameter
    :param choices: Values parameter can take
    """

    def __init__(self, name: str, choices: Tuple[Any, ...]):
        super().__init__(name)
        self._assert_choices(choices)
        self.choices = list(choices)
        self.num_choices = len(self.choices)
        assert self.num_choices > 0

    @staticmethod
    def _assert_value_type(value):
        assert (
            isinstance(value, str) or isinstance(value, int) or isinstance(value, float)
        ), f"value = {value} has type {type(value)}, must be str, int, or float"

    @staticmethod
    def _assert_choices(choices: Tuple[Any, ...]):
        assert len(choices) > 0
        HyperparameterRangeCategorical._assert_value_type(choices[0])
        value_type = type(choices[0])
        assert any(
            type(x) == value_type for x in choices
        ), f"All entries in choices = {choices} must have the same type {value_type}"

    @staticmethod
    def _assert_choices_and_active_choices(
        choices: Tuple[Any, ...], active_choices: Optional[Tuple[Any, ...]] = None
    ) -> Optional[int]:
        HyperparameterRangeCategorical._assert_choices(choices)
        firstpos = None
        if active_choices is not None:
            HyperparameterRangeCategorical._assert_choices(active_choices)
            err_msg = (
                f"active_choices = {active_choices} not contiguous subsequence "
                f"of choices = {choices}"
            )
            try:
                firstpos = choices.index(active_choices[0])
                assert all(
                    a == b for a, b in zip(active_choices, choices[firstpos:])
                ), err_msg
            except ValueError:
                raise AssertionError(err_msg)
        return firstpos

    def __repr__(self) -> str:
        return "{}({}, {})".format(
            self.__class__.__name__, repr(self.name), repr(self.choices)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperparameterRangeCategorical):
            return self.name == other.name and self.choices == other.choices
        return False


class HyperparameterRangeCategoricalNonBinary(HyperparameterRangeCategorical):
    """
    Can take on discrete set of values. We use one-hot encoding internally.
    If the value range has size 2, it is more efficient to use
    :class:`HyperparameterRangeCategoricalBinary`.

    :param name: Name of hyperparameter
    :param choices: Values parameter can take
    :param active_choices: If given, must be nonempty subset of ``choices``.
    """

    def __init__(
        self,
        name: str,
        choices: Tuple[Any, ...],
        active_choices: Tuple[Any, ...] = None,
    ):
        super().__init__(name, choices)
        if active_choices is None:
            if self.num_choices > 1:
                self._ndarray_bounds = [(0.0, 1.0)] * self.num_choices
            else:
                self._ndarray_bounds = [(1.0, 1.0)]
        else:
            self._assert_choices(active_choices)
            _active_choices = set(active_choices)
            num_active_choices = len(active_choices)
            self._ndarray_bounds = [(0.0, 0.0)] * self.num_choices
            num = 0
            val_nonzero = (0.0, 1.0) if num_active_choices > 1 else (1.0, 1.0)
            for pos, val in enumerate(self.choices):
                if val in _active_choices:
                    self._ndarray_bounds[pos] = val_nonzero
                    num += 1
            assert num == num_active_choices, (
                f"active_choices = {active_choices} must be a subset of "
                + f"choices = {choices}"
            )

    def ndarray_size(self) -> int:
        return self.num_choices

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        self._assert_value_type(hp)
        assert hp in self.choices, "{} not in {}".format(hp, self)
        idx = self.choices.index(hp)
        result = np.zeros(shape=(self.num_choices,))
        result[idx] = 1.0
        return result

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        assert len(cand_ndarray) == self.num_choices, (cand_ndarray, self)
        return self.choices[int(np.argmax(cand_ndarray))]

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._ndarray_bounds


class HyperparameterRangeCategoricalBinary(HyperparameterRangeCategorical):
    """
    Here, the value range must be of size 2. The internal encoding is a
    single int, so 1 instead of 2 dimensions.

    :param name: Name of hyperparameter
    :param choices: Values parameter can take (must be size 2)
    :param active_choices: If given, must be nonempty subset of ``choices``.
    """

    def __init__(
        self,
        name: str,
        choices: Tuple[Any, ...],
        active_choices: Tuple[Any, ...] = None,
    ):
        assert len(choices) == 2, (
            f"len(choices) = {len(choices)}, must be 2. Use "
            + "HyperparameterRangeCategoricalNonBinary instead"
        )
        super().__init__(name, choices)
        active_value = None
        if active_choices is not None:
            self._assert_choices(active_choices)
            _active_choices = set(active_choices)
            num = 0
            for pos, val in enumerate(self.choices):
                if val in _active_choices:
                    active_value = pos
                    num += 1
            assert num == len(_active_choices), (
                f"active_choices = {active_choices} must be a subset of "
                + f"choices = {choices}"
            )
            if num == 2:
                active_value = None
        # Internal encoding
        self._range_int = HyperparameterRangeInteger(
            name=name + "_INTERNAL",
            lower_bound=0,
            upper_bound=1,
            scaling=LinearScaling(),
            active_lower_bound=active_value,
            active_upper_bound=active_value,
        )

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        self._assert_value_type(hp)
        assert hp in self.choices, "{} not in {}".format(hp, self)
        idx = self.choices.index(hp)
        return self._range_int.to_ndarray(idx)

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        assert len(cand_ndarray) == 1
        return self.choices[self._range_int.from_ndarray(cand_ndarray)]

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._range_int.get_ndarray_bounds()


class HyperparameterRangeOrdinalEqual(HyperparameterRangeCategorical):
    """
    Ordinal hyperparameter, equal distance encoding. See also
    :class:`~syne_tune.config_space.Ordinal`.

    :param name: Name of hyperparameter
    :param choices: Values parameter can take
    :param active_choices: If given, must be nonempty contiguous
        subsequence of ``choices``.
    """

    def __init__(
        self,
        name: str,
        choices: Tuple[Any, ...],
        active_choices: Optional[Tuple[Any, ...]] = None,
    ):
        super().__init__(name, choices)
        active_lower_bound = self._assert_choices_and_active_choices(
            choices, active_choices
        )
        if active_choices is not None:
            active_upper_bound = active_lower_bound + len(active_choices) - 1
        else:
            active_upper_bound = None
        self._range_int = HyperparameterRangeInteger(
            name=name + "_INTERNAL",
            lower_bound=0,
            upper_bound=self.num_choices - 1,
            scaling=LinearScaling(),
            active_lower_bound=active_lower_bound,
            active_upper_bound=active_upper_bound,
        )

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        self._assert_value_type(hp)
        assert hp in self.choices, "{} not in {}".format(hp, self)
        idx = self.choices.index(hp)
        return self._range_int.to_ndarray(idx)

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        assert len(cand_ndarray) == 1
        return self.choices[self._range_int.from_ndarray(cand_ndarray)]

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._range_int.get_ndarray_bounds()

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperparameterRangeOrdinalEqual):
            return self.name == other.name and self.choices == other.choices
        return False


class HyperparameterRangeOrdinalNearestNeighbor(HyperparameterRangeCategorical):
    """
    Ordinal hyperparameter, nearest neighbour encoding. See also
    :class:`~syne_tune.config_space.OrdinalNearestNeighbor`.

    :param name: Name of hyperparameter
    :param choices: Values parameter can take (numerical values, strictly
        increasing, size ``>= 2``)
    :param log_scale: If ``True``, nearest neighbour done in log (``choices`` must
        be positive)
    :param active_choices: If given, must be nonempty contiguous
        subsequence of ``choices``.
    """

    def __init__(
        self,
        name: str,
        choices: Tuple[Any, ...],
        log_scale: bool = False,
        active_choices: Optional[Tuple[Any, ...]] = None,
    ):
        assert len(choices) > 1, "Use HyperparameterRangeOrdinalEqual"
        super().__init__(name, choices)
        self._domain_int = OrdinalNearestNeighbor(choices, log_scale=log_scale)
        active_lower_bound, active_upper_bound = self._get_active_bounds(active_choices)
        self._range_int = HyperparameterRangeContinuous(
            name=name + "_INTERNAL",
            lower_bound=self._domain_int.lower_int,
            upper_bound=self._domain_int.upper_int,
            scaling=LinearScaling(),
            active_lower_bound=active_lower_bound,
            active_upper_bound=active_upper_bound,
        )

    def _get_active_bounds(
        self, active_choices: Optional[Tuple[Any, ...]]
    ) -> Tuple[Optional[float], Optional[float]]:
        if active_choices is None:
            return None, None
        else:
            di = self._domain_int
            firstpos = self._assert_choices_and_active_choices(
                di.categories, active_choices
            )
            left_thres = di._categories_int[firstpos]
            num_active_choices = len(active_choices)
            if num_active_choices == 1:
                # Fix this value
                active_lower_bound = left_thres
                active_upper_bound = left_thres
            if firstpos > 0:
                diff = left_thres - di._categories_int[firstpos - 1]
                active_lower_bound = left_thres - 0.499 * diff
            else:
                active_lower_bound = di._lower_int
            lastpos = firstpos + num_active_choices - 1
            right_thres = di._categories_int[lastpos]
            if lastpos < len(di) - 1:
                diff = di._categories_int[lastpos + 1] - right_thres
                active_upper_bound = right_thres + 0.499 * diff
            else:
                active_upper_bound = di._upper_int
            return active_lower_bound, active_upper_bound

    @property
    def log_scale(self) -> bool:
        return self._domain_int.log_scale

    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        self._assert_value_type(hp)
        assert hp in self.choices, "{} not in {}".format(hp, self)
        return self._range_int.to_ndarray(
            np.log(float(hp)) if self.log_scale else float(hp)
        )

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        assert len(cand_ndarray) == 1
        return self._domain_int.cast_int(self._range_int.from_ndarray(cand_ndarray))

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._range_int.get_ndarray_bounds()

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperparameterRangeOrdinalNearestNeighbor):
            return (
                self.name == other.name
                and self.choices == other.choices
                and self.log_scale == other.log_scale
            )
        return False


class HyperparameterRangesImpl(HyperparameterRanges):
    """
    Basic implementation of
    :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`.

    :param config_space: Configuration space
    :param name_last_pos: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param value_for_last_pos: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param active_config_space: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param prefix_keys: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        name_last_pos: str = None,
        value_for_last_pos=None,
        active_config_space: Dict[str, Any] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            config_space,
            name_last_pos,
            value_for_last_pos,
            active_config_space,
            prefix_keys,
        )
        hp_ranges = []
        for name in self.internal_keys:
            hp_range = self.config_space[name]
            assert isinstance(hp_range, Domain)
            tp = hp_range.value_type
            if isinstance(hp_range, Categorical):
                kwargs = dict()
                is_in_active = name in self.active_config_space
                num_categories = len(hp_range.categories)
                if is_in_active:
                    kwargs["active_choices"] = tuple(
                        self.active_config_space[name].categories
                    )
                if isinstance(hp_range, OrdinalNearestNeighbor):
                    _cls = HyperparameterRangeOrdinalNearestNeighbor
                    kwargs["log_scale"] = hp_range.log_scale
                elif isinstance(hp_range, Ordinal):
                    _cls = HyperparameterRangeOrdinalEqual
                elif num_categories == 2:
                    _cls = HyperparameterRangeCategoricalBinary
                else:
                    _cls = HyperparameterRangeCategoricalNonBinary
                hp_ranges.append(
                    _cls(
                        name,
                        choices=tuple(hp_range.categories),
                        **kwargs,
                    )
                )
            else:
                scaling = get_scaling(hp_range)
                kwargs = {
                    "name": name,
                    "lower_bound": hp_range.lower,
                    "upper_bound": hp_range.upper,
                    "scaling": scaling,
                }
                if isinstance(hp_range, FiniteRange):
                    assert (
                        name not in self.active_config_space
                    ), f"Parameter '{name}' of type FiniteRange cannot be used in active_config_space"
                    hp_ranges.append(
                        HyperparameterRangeFiniteRange(
                            **kwargs, size=len(hp_range), cast_int=hp_range.cast_int
                        )
                    )
                else:
                    if name in self.active_config_space:
                        active_hp_range = self.active_config_space[name]
                        kwargs.update(
                            {
                                "active_lower_bound": active_hp_range.lower,
                                "active_upper_bound": active_hp_range.upper,
                            }
                        )
                    if tp == float:
                        hp_ranges.append(HyperparameterRangeContinuous(**kwargs))
                    else:
                        hp_ranges.append(HyperparameterRangeInteger(**kwargs))
        self._hp_ranges = hp_ranges
        csum = [0] + list(np.cumsum([d.ndarray_size() for d in hp_ranges]))
        self._ndarray_size = csum[-1]
        self._encoded_ranges = dict(
            zip((d.name for d in hp_ranges), zip(csum[:-1], csum[1:]))
        )

    @property
    def ndarray_size(self) -> int:
        return self._ndarray_size

    def to_ndarray(self, config: Configuration) -> np.ndarray:
        config_tpl = self.config_to_tuple(config)
        pieces = [
            hp_range.to_ndarray(hp) for hp_range, hp in zip(self._hp_ranges, config_tpl)
        ]
        return np.hstack(pieces)

    def from_ndarray(self, enc_config: np.ndarray) -> Configuration:
        enc_config = enc_config.reshape((-1, 1))
        assert enc_config.size == self._ndarray_size, (
            enc_config.size,
            self._ndarray_size,
        )
        hps = []
        start = 0
        for hp_range in self._hp_ranges:
            end = start + hp_range.ndarray_size()
            enc_attr = enc_config[start:end]
            hps.append(hp_range.from_ndarray(enc_attr))
            start = end
        return self.tuple_to_config(tuple(hps))

    @property
    def encoded_ranges(self) -> Dict[str, Tuple[int, int]]:
        return self._encoded_ranges

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            x for hp_range in self._hp_ranges for x in hp_range.get_ndarray_bounds()
        ]
        if self.is_attribute_fixed():
            hp_range = self._hp_ranges[-1]
            assert hp_range.name == self.name_last_pos
            enc_fixed = hp_range.to_ndarray(self.value_for_last_pos).reshape((-1,))
            offset = self.ndarray_size - enc_fixed.size
            for i, val in enumerate(enc_fixed):
                bounds[i + offset] = (val, val)
        return bounds

    def __repr__(self) -> str:
        return "{}{}".format(self.__class__.__name__, repr(self._hp_ranges))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangesImpl):
            return self._hp_ranges == other._hp_ranges
        return False


def decode_extended_features(
    features_ext: np.ndarray,
    resource_attr_range: Tuple[int, int],
) -> (np.ndarray, np.ndarray):
    """
    Given matrix of features from extended configs, corresponding to
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext.ExtendedConfiguration`,
    split into feature matrix from normal configs and resource values.

    :param features_ext: Matrix of features from extended configs
    :param resource_attr_range: ``(r_min, r_max)``
    :return: ``(features, resources)``
    """
    r_min, r_max = resource_attr_range
    features = features_ext[:, :-1]
    resources_encoded = features_ext[:, -1].reshape((-1,))
    lower = r_min - 0.5 + EPS
    width = r_max - r_min + 1 - 2 * EPS
    resources = np.clip(
        np.round(resources_encoded * width + lower), r_min, r_max
    ).astype("int64")
    return features, resources
