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

# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.
import logging
from copy import copy
from inspect import signature
from math import isclose
import sys
from typing import Any, Callable, List, Optional, Sequence, Union
import argparse

import numpy as np

from syne_tune.util import is_increasing

logger = logging.getLogger(__name__)


class Domain:
    """Base class to specify a type and valid range to sample parameters from.

    This base class is implemented by parameter spaces, like float ranges
    (``Float``), integer ranges (``Integer``), or categorical variables
    (``Categorical``). The ``Domain`` object contains information about
    valid values (e.g. minimum and maximum values), and exposes methods that
    allow specification of specific samplers (e.g. ``uniform()`` or
    ``loguniform()``).

    """

    sampler = None
    default_sampler_cls = None

    @property
    def value_type(self):
        raise NotImplementedError

    def cast(self, value):
        """Cast value to domain type"""
        return self.value_type(value)

    def set_sampler(self, sampler, allow_override=False):
        if self.sampler and not allow_override:
            raise ValueError(
                "You can only choose one sampler for parameter "
                "domains. Existing sampler for parameter {}: "
                "{}. Tried to add {}".format(
                    self.__class__.__name__, self.sampler, sampler
                )
            )
        self.sampler = sampler

    def get_sampler(self) -> "Sampler":
        sampler = self.sampler
        if not sampler:
            sampler = self.default_sampler_cls()
        return sampler

    def sample(self, spec=None, size=1, random_state=None):
        """
        :param size: Number of values to sample
        :param random_state: PRN generator
        :return: Single value (`size == 1`) or list (`size > 1`)
        """
        sampler = self.get_sampler()
        return sampler.sample(self, spec=spec, size=size, random_state=random_state)

    def is_grid(self):
        return isinstance(self.sampler, Grid)

    def is_function(self):
        return False

    def is_valid(self, value: Any):
        """Returns True if `value` is a valid value in this domain."""
        raise NotImplementedError

    @property
    def domain_str(self):
        return "(unknown)"

    def __len__(self):
        """
        :return: Size of domain (number of distinct elements), or 0 if size
            is infinite
        """
        raise NotImplementedError

    def match_string(self, value) -> str:
        """
        Returns string representation of `value` (which must be of domain type)
        which is to match configurations for (approximate) equality.
        For discrete types (e.g., `Integer`, `Categorical`), this matches for
        exact equality.

        :param value: Value of domain type (use `cast` to be safe)
        :return: String representation useful for matching
        """
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if self.sampler is None:
            return other.sampler is None
        else:
            return self.sampler == other.sampler


class Sampler:
    def sample(
        self,
        domain: Domain,
        spec: Optional[Union[List[dict], dict]] = None,
        size: int = 1,
        random_state: Optional[np.random.RandomState] = None,
    ):
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        raise NotImplementedError


class BaseSampler(Sampler):
    def __str__(self):
        return "Base"


class Uniform(Sampler):
    def __str__(self):
        return "Uniform"

    def __eq__(self, other) -> bool:
        return isinstance(other, Uniform)


EXP_ONE = np.exp(1.0)


class LogUniform(Sampler):
    """
    Note: We keep the argument `base` for compatibility with Ray Tune.
    Since `base` has no effect on the distribution, we don't use it
    internally.

    """

    def __init__(self, base: float = EXP_ONE):
        assert base > 0, "Base has to be strictly greater than 0"
        self.base = base  # Not really used internally

    def __str__(self):
        return "LogUniform"

    def __eq__(self, other) -> bool:
        return isinstance(other, LogUniform) and self.base == other.base


class Normal(Sampler):
    def __init__(self, mean: float = 0.0, sd: float = 0.0):
        self.mean = mean
        self.sd = sd

        assert self.sd > 0, "SD has to be strictly greater than 0"

    def __str__(self):
        return "Normal"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Normal)
            and np.isclose(self.mean, other.mean)
            and np.isclose(self.sd, other.sd)
        )


class Grid(Sampler):
    """Dummy sampler used for grid search"""

    def sample(
        self,
        domain: Domain,
        spec: Optional[Union[List[dict], dict]] = None,
        size: int = 1,
        random_state: Optional[np.random.RandomState] = None,
    ):
        return RuntimeError("Do not call `sample()` on grid.")

    def __eq__(self, other) -> bool:
        return isinstance(other, Grid)


def _sanitize_sample_result(items, domain: Domain):
    if len(items) > 1:
        return [domain.cast(x) for x in items]
    else:
        return domain.cast(items[0])


class Float(Domain):
    class _Uniform(Uniform):
        def sample(
            self,
            domain: "Float",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            assert domain.lower > float("-inf"), "Uniform needs a lower bound"
            assert domain.upper < float("inf"), "Uniform needs a upper bound"
            if random_state is None:
                random_state = np.random
            items = random_state.uniform(domain.lower, domain.upper, size=size)
            return _sanitize_sample_result(items, domain)

    class _LogUniform(LogUniform):
        def sample(
            self,
            domain: "Float",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            assert domain.lower > 0, "LogUniform needs a lower bound greater than 0"
            assert (
                0 < domain.upper < float("inf")
            ), "LogUniform needs a upper bound greater than 0"
            # Note: We don't use `self.base` here, because it does not make a
            # difference
            logmin = np.log(domain.lower)
            logmax = np.log(domain.upper)
            if random_state is None:
                random_state = np.random
            log_items = random_state.uniform(logmin, logmax, size=size)
            items = np.exp(log_items)
            return _sanitize_sample_result(items, domain)

    # Transform is -log(1 - x)
    class _ReverseLogUniform(LogUniform):
        def sample(
            self,
            domain: "Float",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            assert 0 <= domain.lower <= domain.upper < 1
            logmin = -np.log1p(-domain.lower)
            logmax = -np.log1p(-domain.upper)
            if random_state is None:
                random_state = np.random
            log_items = random_state.uniform(logmin, logmax, size=size)
            items = -np.expm1(-log_items)
            return _sanitize_sample_result(items, domain)

    class _Normal(Normal):
        def sample(
            self,
            domain: "Float",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            assert not domain.lower or domain.lower == float(
                "-inf"
            ), "Normal sampling does not allow a lower value bound."
            assert not domain.upper or domain.upper == float(
                "inf"
            ), "Normal sampling does not allow a upper value bound."
            if random_state is None:
                random_state = np.random
            items = random_state.normal(self.mean, self.sd, size=size)
            return _sanitize_sample_result(items, domain)

    default_sampler_cls = _Uniform

    def __init__(self, lower: Optional[float], upper: Optional[float]):
        # Need to explicitly check for None
        self.lower = lower if lower is not None else float("-inf")
        self.upper = upper if upper is not None else float("inf")

    @property
    def value_type(self):
        return float

    def uniform(self):
        if not self.lower > float("-inf"):
            raise ValueError(
                "Uniform requires a lower bound. Make sure to set the "
                "`lower` parameter of `Float()`."
            )
        if not self.upper < float("inf"):
            raise ValueError(
                "Uniform requires a upper bound. Make sure to set the "
                "`upper` parameter of `Float()`."
            )
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def loguniform(self):
        if not self.lower > 0:
            raise ValueError(
                "LogUniform requires a lower bound greater than 0."
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead."
            )
        if not 0 < self.upper < float("inf"):
            raise ValueError(
                "LogUniform requires a upper bound greater than 0. "
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead."
            )
        new = copy(self)
        new.set_sampler(self._LogUniform())
        return new

    def reverseloguniform(self):
        if not (0 <= self.lower <= self.upper < 1):
            raise ValueError(
                "ReverseLogUniform requires 0 <= lower <= upper < 1."
                f"Got: lower={self.lower}, upper={self.upper}. Did you "
                "pass a variable that has been transformed as -log(1 - x)?"
                "If so, pass the non-transformed values instead."
            )
        new = copy(self)
        new.set_sampler(self._ReverseLogUniform())
        return new

    def normal(self, mean=0.0, sd=1.0):
        new = copy(self)
        new.set_sampler(self._Normal(mean, sd))
        return new

    def quantized(self, q: float):
        if self.lower > float("-inf") and not isclose(
            self.lower / q, round(self.lower / q)
        ):
            raise ValueError(
                f"Your lower variable bound {self.lower} is not divisible by "
                f"quantization factor {q}."
            )
        if self.upper < float("inf") and not isclose(
            self.upper / q, round(self.upper / q)
        ):
            raise ValueError(
                f"Your upper variable bound {self.upper} is not divisible by "
                f"quantization factor {q}."
            )

        new = copy(self)
        new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
        return new

    def is_valid(self, value: float):
        return self.lower <= value <= self.upper

    @property
    def domain_str(self):
        return f"({self.lower}, {self.upper})"

    def __len__(self):
        if self.lower < self.upper:
            return 0
        else:
            return 1

    def match_string(self, value) -> str:
        return f"{value:.7e}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Float)
            and super(Float, self).__eq__(other)
            and np.isclose(self.lower, other.lower)
            and np.isclose(self.upper, other.upper)
        )

    def __repr__(self):
        name = "loguniform" if is_log_space(self) else "uniform"
        return name + self.domain_str


class Integer(Domain):
    class _Uniform(Uniform):
        def sample(
            self,
            domain: "Integer",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            if random_state is None:
                random_state = np.random
            # Note: domain.upper is inclusive here, but exclusive in
            # `np.random.randint`.
            items = random_state.randint(domain.lower, domain.upper + 1, size=size)
            return _sanitize_sample_result(items, domain)

    class _LogUniform(LogUniform):
        def sample(
            self,
            domain: "Integer",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            assert domain.lower > 0, "LogUniform needs a lower bound greater than 0"
            assert (
                0 < domain.upper < float("inf")
            ), "LogUniform needs a upper bound greater than 0"
            # Note: We don't use `self.base` here, because it does not make a
            # difference
            logmin = np.log(domain.lower)
            logmax = np.log(domain.upper)
            if random_state is None:
                random_state = np.random
            log_items = random_state.uniform(logmin, logmax, size=size)
            items = np.exp(log_items)
            items = np.round(items).astype(int)
            return _sanitize_sample_result(items, domain)

    default_sampler_cls = _Uniform

    def __init__(self, lower, upper):
        self.lower = self.cast(lower)
        self.upper = self.cast(upper)

    @property
    def value_type(self):
        return int

    def cast(self, value):
        return int(round(value))

    def quantized(self, q: int):
        new = copy(self)
        new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
        return new

    def uniform(self):
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def loguniform(self):
        if not self.lower > 0:
            raise ValueError(
                "LogUniform requires a lower bound greater than 0."
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead."
            )
        if not 0 < self.upper < float("inf"):
            raise ValueError(
                "LogUniform requires a upper bound greater than 0. "
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead."
            )
        new = copy(self)
        new.set_sampler(self._LogUniform())
        return new

    def is_valid(self, value: int):
        return self.lower <= value <= self.upper

    @property
    def domain_str(self):
        return f"({self.lower}, {self.upper})"

    def __len__(self):
        return self.upper - self.lower + 1

    def match_string(self, value) -> str:
        return str(value)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Integer)
            and super(Integer, self).__eq__(other)
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def __repr__(self):
        name = "lograndint" if is_log_space(self) else "randint"
        return name + self.domain_str


class Categorical(Domain):
    class _Uniform(Uniform):
        def sample(
            self,
            domain: "Categorical",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            if random_state is None:
                random_state = np.random
            categories = domain.categories
            items = [
                categories[i] for i in random_state.choice(len(categories), size=size)
            ]
            return _sanitize_sample_result(items, domain)

    default_sampler_cls = _Uniform

    def __init__(self, categories: Sequence):
        assert len(categories) > 0
        self.categories = list(categories)
        value_type = self.value_type
        assert all(
            type(x) == value_type for x in self.categories
        ), f"All entries in categories = {self.categories} must have the same type"
        if isinstance(self.value_type, float):
            logger.warning(
                "The configuration space contains a categorical value with float type. "
                "When performing remote execution, floats are converted to string which can cause rounding "
                "issues. In case of problem, consider using string to represent the float."
            )

    def uniform(self):
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def grid(self):
        new = copy(self)
        new.set_sampler(Grid())
        return new

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, item):
        return self.categories[item]

    def is_valid(self, value: Any):
        return value in self.categories

    @property
    def value_type(self):
        return type(self.categories[0])

    @property
    def domain_str(self):
        return f"{self.categories}"

    def cast(self, value):
        value = self.value_type(value)
        if value not in self.categories:
            assert isinstance(
                value, float
            ), f"value = {value} not contained in categories = {self.categories}"
            # For value type float, we do nearest neighbor matching, in order to
            # avoid meaningless mistakes due to round-off or conversions from
            # string and back
            categ_arr = np.array(self.categories)
            distances = np.abs(categ_arr - value)
            minind = np.argmin(distances)
            assert distances[minind] < 0.01 * abs(
                categ_arr[minind]
            ), f"value = {value} not contained or close to any in categories = {self.categories}"
            value = self.categories[minind]
        return value

    def match_string(self, value) -> str:
        return str(self.categories.index(value))

    def __repr__(self):
        return f"choice({self.categories})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Categorical)
            and super(Categorical, self).__eq__(other)
            and self.categories == other.categories
        )


class Ordinal(Categorical):
    """
    Represents an ordered set. As far as random sampling is concerned, this
    type is equivalent to :class:`Categorical`, but when used in methods
    that require encodings (or distances), nearby values have closer
    encodings.
    """

    def __init__(self, categories: Sequence):
        super().__init__(categories)

    def __repr__(self):
        return f"ordinal({self.categories}, kind='equal')"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Ordinal)
            and super(Ordinal, self).__eq__(other)
            and self.categories == other.categories
        )


class OrdinalNearestNeighbor(Ordinal):
    """
    Different type for ordered set of numerical values (int or float).
    Essentially, the finite set is represented by a real-valued interval
    containing all values, and random sampling draws a value from this
    interval and rounds it to the nearest value in `categories`. If
    `log_scale` is True, all of this happens in log scale. Unless values
    are equidistant, this is different from `Ordinal`.
    """

    def __init__(self, categories: Sequence, log_scale: bool = False):
        super().__init__(categories)
        self.log_scale = log_scale
        self._initialize()

    def __repr__(self):
        if self.log_scale:
            return f"logordinal({self.categories})"
        else:
            return f"ordinal({self.categories}, kind='nn')"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, OrdinalNearestNeighbor)
            and super(Ordinal, self).__eq__(other)
            and self.categories == other.categories
            and self.log_scale == other.log_scale
        )

    def _initialize(self):
        assert (
            self.value_type == int or self.value_type == float
        ), f"Value type must be int or float: {self.categories}"
        assert is_increasing(
            self.categories
        ), f"Values must be strictly increasing: {self.categories}"
        assert (
            not self.log_scale or self.categories[0] > 0
        ), f"Values must be positive: {self.categories}"
        self._lower_int = None
        self._upper_int = None
        self._categories_int = None
        self._more_than_one_category = len(self.categories) > 1
        if self._more_than_one_category:
            if self.log_scale:
                self._categories_int = np.array(
                    [np.log(float(x)) for x in self.categories]
                )
            else:
                self._categories_int = np.array([float(x) for x in self.categories])
            avg_dist = 0.5 * np.mean(
                self._categories_int[1:] - self._categories_int[:-1]
            )
            self._lower_int = self._categories_int[0] - avg_dist
            self._upper_int = self._categories_int[-1] + avg_dist

    @property
    def lower_int(self) -> Optional[float]:
        return self._lower_int

    @property
    def upper_int(self) -> Optional[float]:
        return self._upper_int

    @property
    def categories_int(self) -> Optional[np.ndarray]:
        return self._categories_int

    def cast_int(self, value_int: float):
        if self._more_than_one_category:
            distances = np.abs(self._categories_int - value_int)
            minind = np.argmin(distances)
        else:
            minind = 0
        return self.categories[minind]

    def cast(self, value):
        return self.cast_int(np.log(float(value)) if self.log_scale else float(value))

    def set_sampler(self, sampler, allow_override=False):
        raise NotImplementedError()

    def get_sampler(self):
        return None

    def sample(self, spec=None, size=1, random_state=None):
        if random_state is None:
            random_state = np.random
        items = random_state.uniform(self._lower_int, self._upper_int, size=size)
        if size > 1:
            return [self.cast_int(x) for x in items]
        else:
            return self.cast_int(items)


class Function(Domain):
    class _CallSampler(BaseSampler):
        def sample(
            self,
            domain: "Function",
            spec: Optional[Union[List[dict], dict]] = None,
            size: int = 1,
            random_state: Optional[np.random.RandomState] = None,
        ):
            if random_state is not None:
                raise NotImplementedError()
            if domain.pass_spec:
                items = [
                    domain.func(spec[i] if isinstance(spec, list) else spec)
                    for i in range(size)
                ]
            else:
                items = [domain.func() for i in range(size)]

            return _sanitize_sample_result(items, domain)

    default_sampler_cls = _CallSampler

    def __init__(self, func: Callable):
        sig = signature(func)

        pass_spec = True  # whether we should pass `spec` when calling `func`
        try:
            sig.bind({})
        except TypeError:
            pass_spec = False

        if not pass_spec:
            try:
                sig.bind()
            except TypeError as exc:
                raise ValueError(
                    "The function passed to a `Function` parameter must be "
                    "callable with either 0 or 1 parameters."
                ) from exc

        self.pass_spec = pass_spec
        self.func = func

    def is_function(self):
        return True

    def is_valid(self, value: Any):
        return True  # This is user-defined, so lets not assume anything

    @property
    def domain_str(self):
        return f"{self.func}()"

    def __len__(self):
        return 0


class Quantized(Sampler):
    def __init__(self, sampler: Sampler, q: Union[float, int]):
        self.sampler = sampler
        self.q = q

        assert self.sampler, "Quantized() expects a sampler instance"

    def get_sampler(self):
        return self.sampler

    def sample(
        self,
        domain: Domain,
        spec: Optional[Union[List[dict], dict]] = None,
        size: int = 1,
        random_state: Optional[np.random.RandomState] = None,
    ):
        values = self.sampler.sample(domain, spec, size, random_state)
        quantized = np.round(np.divide(values, self.q)) * self.q
        if not isinstance(quantized, np.ndarray):
            return domain.cast(quantized)
        return list(quantized)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Quantized)
            and self.q == other.q
            and self.sampler == other.sampler
        )


class FiniteRange(Domain):
    """
    Represents a finite range `[lower, ..., upper]` with `size` values
    equally spaced in linear or log domain.
    If `cast_int`, the value type is int (rounding after the transform).
    """

    def __init__(
        self,
        lower: float,
        upper: float,
        size: int,
        log_scale: bool = False,
        cast_int: bool = False,
    ):
        assert lower <= upper
        assert size >= 1
        if log_scale:
            assert lower > 0.0
        self._uniform_int = randint(0, size - 1)
        self.lower = lower
        self.upper = upper
        self.log_scale = log_scale
        self.cast_int = cast_int
        self.size = size
        if not log_scale:
            self._lower_internal = lower
            self._step_internal = (upper - lower) / (size - 1) if size > 1 else 0
        else:
            self._lower_internal = np.log(lower)
            upper_internal = np.log(upper)
            self._step_internal = (
                (upper_internal - self._lower_internal) / (size - 1) if size > 1 else 0
            )
        self._values = [self._map_from_int(x) for x in range(self.size)]

    @property
    def values(self):
        return self._values

    def _map_from_int(self, x: int) -> Union[float, int]:
        y = x * self._step_internal + self._lower_internal
        if self.log_scale:
            y = np.exp(y)
        res = float(np.clip(y, self.lower, self.upper))
        if self.cast_int:
            res = int(np.rint(res))
        return res

    def __repr__(self):
        values_str = ",".join([str(x) for x in self._values])
        name = "logfinrange" if self.log_scale else "finrange"
        return f"{name}([{values_str}])"

    @property
    def value_type(self):
        return float if not self.cast_int else int

    def _map_to_int(self, value) -> int:
        if self._step_internal == 0:
            return 0
        else:
            int_value = np.clip(value, self.lower, self.upper)
            if self.log_scale:
                int_value = np.log(int_value)
            sz = len(self._uniform_int)
            return int(
                np.clip(
                    round((int_value - self._lower_internal) / self._step_internal),
                    0,
                    sz - 1,
                )
            )

    def cast(self, value):
        return self._values[self._map_to_int(value)]

    def set_sampler(self, sampler, allow_override=False):
        raise NotImplementedError()

    def get_sampler(self):
        return None

    def sample(self, spec=None, size=1, random_state=None):
        int_sample = self._uniform_int.sample(spec, size, random_state)
        if size > 1:
            return [self._values[x] for x in int_sample]
        else:
            return self._values[int_sample]

    @property
    def domain_str(self):
        return f"({self.lower}, {self.upper}, {self.__len__()})"

    def __len__(self):
        return len(self._uniform_int)

    def match_string(self, value) -> str:
        return str(self._map_to_int(value))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FiniteRange)
            and np.isclose(self.lower, other.lower)
            and np.isclose(self.upper, other.upper)
            and self.log_scale == other.log_scale
            and self.cast_int == other.cast_int
        )


def sample_from(func: Callable[[dict], Any]):
    """Specify that tune should sample configuration values from this function.

    Arguments:
        func: An callable function to draw a sample from.
    """
    return Function(func)


def uniform(lower: float, upper: float):
    """Sample a float value uniformly between ``lower`` and ``upper``.

    Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
    ``np.random.uniform(1, 10))``

    """
    return Float(lower, upper).uniform()


def quniform(lower: float, upper: float, q: float):
    """Sample a quantized float value uniformly between ``lower`` and ``upper``.

    Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
    ``np.random.uniform(1, 10))``

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Float(lower, upper).uniform().quantized(q)


def loguniform(lower: float, upper: float):
    """Sugar for sampling in different orders of magnitude.

    Note: Ray Tune has an argument `base` here, but since this does not
    affect the distribution, we drop it.

    Args:
        lower (float): Lower boundary of the output interval (e.g. 1e-4)
        upper (float): Upper boundary of the output interval (e.g. 1e-2)

    """
    return Float(lower, upper).loguniform()


def reverseloguniform(lower: float, upper: float):
    """Values 0 <= x < 1, internally represented as -log(1 - x)

    Args:
        lower (float): Lower boundary of the output interval (e.g. 0.99)
        upper (float): Upper boundary of the output interval (e.g. 0.9999)

    """
    return Float(lower, upper).reverseloguniform()


def qloguniform(lower: float, upper: float, q: float):
    """Sugar for sampling in different orders of magnitude.

    The value will be quantized, i.e. rounded to an integer increment of ``q``.

    Quantization makes the upper bound inclusive.

    Args:
        lower (float): Lower boundary of the output interval (e.g. 1e-4)
        upper (float): Upper boundary of the output interval (e.g. 1e-2)
        q (float): Quantization number. The result will be rounded to an
            integer increment of this value.

    """
    return Float(lower, upper).loguniform().quantized(q)


def choice(categories: list):
    """Sample a categorical value.

    Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
    ``random.choice([1, 2])``

    """
    return Categorical(categories).uniform()


def ordinal(categories: list, kind: Optional[str] = None):
    """
    Ordinal value from list `categories`. Different variants are selected by
    `kind`.

    For `kind == "equal"`, sampling is the same as for `choice`, and the
    internal encoding is by int (first value maps to 0, second to 1, ...).

    For `kind == "nn"`, the finite set is represented by a real-valued interval
    containing all values, and random sampling draws a value from this
    interval and rounds it to the nearest value in `categories`. This behaves
    like a finite version of `uniform` or `randint`. For `kind == "nn-log"`,
    nearest neighbour rounding happens in log space, which behaves like a
    finite version of `loguniform` or `lograndint`. You can also use the
    synonym `logordinal`.
    For this type, values in `categories` must be int or float and strictly
    increasing, and also positive if `kind == "nn-log"`.

    :param categories:
    :param kind: Can be "equal", "nn", "nn-log"
    """
    if kind is None:
        # Default is "nn" for value type float or int and increasing,
        # "equal" otherwise
        kind = "equal"
        if len(categories) > 1 and isinstance(categories[0], (int, float)):
            if is_increasing(categories):
                kind = "nn"
            else:
                logger.info(
                    "Using kind='equal' for ordinal, since categories are not sorted in increasing order"
                )
    if kind == "equal":
        return Ordinal(categories).uniform()
    else:
        log_scale = kind == "nn-log"
        assert log_scale or kind == "nn", f"kind = {kind} not supported"
        return OrdinalNearestNeighbor(categories, log_scale=log_scale)


def logordinal(categories: list):
    """
    Ordinal value from list `categories`.

    Corresponds to `ordinal` with `kind="nn-log"`, so that nearest neighbour
    mapping happens in log scale. Values in `categories` must be int or
    float, strictly increasing, and positive.

    :param categories:
    """
    return OrdinalNearestNeighbor(categories, log_scale=True)


def randint(lower: int, upper: int):
    """Sample an integer value uniformly between ``lower`` and ``upper``.

    ``lower`` and ``upper`` are inclusive. This is a difference to Ray Tune,
    where ``upper`` is exclusive. However, both `lograndint` and `qrandint`
    have inclusive ``upper`` in Ray Tune, so we fix this inconsistency here.

    Sampling from ``tune.randint(10)`` is equivalent to sampling from
    ``np.random.randint(10 + 1)``.

    """
    return Integer(lower, upper).uniform()


def lograndint(lower: int, upper: int):
    """Sample an integer value log-uniformly between ``lower`` and ``upper``

    ``lower`` and ``upper` are inclusive.

    Note: Ray Tune has an argument `base` here, but since this does not
    affect the distribution, we drop it.
    """
    return Integer(lower, upper).loguniform()


def qrandint(lower: int, upper: int, q: int = 1):
    """Sample an integer value uniformly between ``lower`` and ``upper``.

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Integer(lower, upper).uniform().quantized(q)


def qlograndint(lower: int, upper: int, q: int):
    """Sample an integer value log-uniformly between ``lower`` and ``upper``

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Integer(lower, upper).loguniform().quantized(q)


def randn(mean: float = 0.0, sd: float = 1.0):
    """Sample a float value normally with ``mean`` and ``sd``.

    Args:
        mean (float): Mean of the normal distribution. Defaults to 0.
        sd (float): SD of the normal distribution. Defaults to 1.

    """
    return Float(None, None).normal(mean, sd)


def qrandn(mean: float, sd: float, q: float):
    """Sample a float value normally with ``mean`` and ``sd``.

    The value will be quantized, i.e. rounded to an integer increment of ``q``.

    Args:
        mean (float): Mean of the normal distribution.
        sd (float): SD of the normal distribution.
        q (float): Quantization number. The result will be rounded to an
            integer increment of this value.

    """
    return Float(None, None).normal(mean, sd).quantized(q)


def finrange(lower: float, upper: float, size: int, cast_int: bool = False):
    """
    Finite range `[lower, ..., upper]` with `size` entries, which are
    equi-spaced. Finite alternative to `uniform`.

    :param lower: Smallest feasible value
    :param upper: Largest feasible value
    :param size: Size of (finite) domain, must be >= 2
    :param cast_int: Values rounded to int?
    """
    return FiniteRange(lower, upper, size, log_scale=False, cast_int=cast_int)


def logfinrange(lower: float, upper: float, size: int, cast_int: bool = False):
    """
    Finite range `[lower, ..., upper]` with `size` entries, which are
    equi-spaced in the log domain. Finite alternative to `loguniform`.

    :param lower: Smallest feasible value (positive)
    :param upper: Largest feasible value (positive)
    :param size: Size of (finite) domain, must be >= 2
    :param cast_int: Values rounded to int?
    """
    return FiniteRange(lower, upper, size, log_scale=True, cast_int=cast_int)


def is_log_space(domain: Domain) -> bool:
    if isinstance(domain, FiniteRange):
        return domain.log_scale
    elif isinstance(domain, OrdinalNearestNeighbor):
        return domain.log_scale
    else:
        sampler = domain.get_sampler()
        return isinstance(sampler, Float._LogUniform) or isinstance(
            sampler, Integer._LogUniform
        )


def is_reverse_log_space(domain: Domain) -> bool:
    return isinstance(domain, Float) and isinstance(
        domain.get_sampler(), Float._ReverseLogUniform
    )


def is_uniform_space(domain: Domain) -> bool:
    if isinstance(domain, FiniteRange):
        return not domain.log_scale
    else:
        sampler = domain.get_sampler()
        return isinstance(sampler, Float._Uniform) or isinstance(
            sampler, Integer._Uniform
        )


def add_to_argparse(parser: argparse.ArgumentParser, config_space: dict):
    """
    Use this to prepare argument parser in endpoint script, for the
    non-fixed parameters in `config_space`.

    :param parser:
    :param config_space:
    :return:
    """
    for name, domain in config_space.items():
        tp = domain.value_type if isinstance(domain, Domain) else type(domain)
        parser.add_argument(f"--{name}", type=tp, required=True)


def cast_config_values(config: dict, config_space: dict) -> dict:
    """
    Returns config with keys, values of `config`, but values are casted to
    their specific types.

    :param config: Config whose values are to be casted
    :param config_space:
    :return: New config with values casted to correct types
    """
    return {
        name: domain.cast(config[name]) if isinstance(domain, Domain) else config[name]
        for name, domain in config_space.items()
        if name in config
    }


def non_constant_hyperparameter_keys(config_space: dict) -> List[str]:
    """
    :param config_space:
    :return: Keys corresponding to (non-fixed) hyperparameters
    """
    return [name for name, domain in config_space.items() if isinstance(domain, Domain)]


def config_space_size(config_space: dict, upper_limit: int = 2**20) -> Optional[int]:
    """
    Counts the number of distinct configurations in the configuration space
    `config_space`. If this is infinite (due to real-valued parameters) or
    larger than `upper_limit`, None is returned.
    """
    assert upper_limit > 1
    size = 1
    for name, domain in config_space.items():
        if isinstance(domain, Domain):
            domain_size = len(domain)
            if domain_size == 0 or domain_size > upper_limit:
                return None  # Try to avoid overflow
            size *= domain_size
            if size > upper_limit:
                return None
    return size


def config_to_match_string(config: dict, config_space: dict, keys: List[str]) -> str:
    """
    Maps configuration to a match string, which can be used to compare configs
    for (approximate) equality. Only keys in `keys` are used, in that ordering.

    :param config: Configuration to be encoded in match string
    :param config_space: Configuration space
    :param keys: Keys of parameters to be encoded
    :return: Match string
    """
    parts = []
    for key in keys:
        domain = config_space[key]
        value = config[key]
        parts.append(f"{key}:{domain.match_string(value)}")
    return ",".join(parts)


def to_dict(x: "Domain") -> dict:
    """
    We assume that for each `Domain` subclass, the `__init__` kwargs are
    also members, and all other members start with `_`.

    """
    domain_kwargs = {
        k: v for k, v in x.__dict__.items() if k != "sampler" and not k.startswith("_")
    }
    result = {
        "domain_cls": x.__class__.__name__,
        "domain_kwargs": domain_kwargs,
    }
    sampler = x.get_sampler()
    if sampler is not None:
        result.update({"sampler_cls": str(sampler), "sampler_kwargs": sampler.__dict__})
    return result


def from_dict(d: dict) -> Domain:
    domain_cls = getattr(sys.modules[__name__], d["domain_cls"])
    domain_kwargs = d["domain_kwargs"]
    domain = domain_cls(**domain_kwargs)
    if "sampler_cls" in d:
        sampler_cls = getattr(domain_cls, "_" + d["sampler_cls"])
        sampler_kwargs = d["sampler_kwargs"]
        sampler = sampler_cls(**sampler_kwargs)
        domain.set_sampler(sampler)
    return domain


def restrict_domain(numerical_domain: Domain, lower: float, upper: float) -> Domain:
    """
    Restricts a numerical domain to be in the range [lower, upper]
    :return:
    """
    assert hasattr(numerical_domain, "lower") and hasattr(numerical_domain, "upper")
    lower = numerical_domain.cast(lower)
    upper = numerical_domain.cast(upper)
    assert lower <= upper
    if not isinstance(numerical_domain, FiniteRange):
        # domain is numerical, set new lower and upper ranges with bounding-box values
        new_domain_dict = to_dict(numerical_domain)
        new_domain_dict["domain_kwargs"]["lower"] = lower
        new_domain_dict["domain_kwargs"]["upper"] = upper
        return from_dict(new_domain_dict)
    else:
        values = numerical_domain.values
        assert lower <= max(numerical_domain._values)
        assert upper >= min(numerical_domain._values)
        i = 0
        while values[i] < lower and i < len(values) - 1:
            i += 1
        new_lower = values[i]

        j = len(values) - 1
        while upper < values[j] and i < j:
            j -= 1
        new_upper = values[j]
        return FiniteRange(
            lower=new_lower,
            upper=new_upper,
            size=j - i + 1,
            cast_int=numerical_domain.cast_int,
            log_scale=numerical_domain.log_scale,
        )
