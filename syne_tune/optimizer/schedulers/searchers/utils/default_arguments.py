from typing import Set, Tuple, Dict, Optional, Any
import logging
import numbers

logger = logging.getLogger(__name__)


class CheckType:
    def assert_valid(self, key: str, value):
        pass


class Float(CheckType):
    def __init__(self, lower: float = None, upper: float = None):
        if lower and upper:
            assert lower < upper
        self.lower = lower
        self.upper = upper

    def assert_valid(self, key: str, value):
        assert isinstance(
            value, numbers.Real
        ), "{}: Value = {} must be of type float".format(key, value)
        assert (
            not self.lower
        ) or value >= self.lower, "{}: Value = {} must be >= {}".format(
            key, value, self.lower
        )
        assert (
            not self.upper
        ) or value <= self.upper, "{}: Value = {} must be <= {}".format(
            key, value, self.upper
        )


class Integer(CheckType):
    def __init__(self, lower: int = None, upper: int = None):
        if lower and upper:
            assert lower < upper
        self.lower = lower
        self.upper = upper

    def assert_valid(self, key: str, value):
        assert isinstance(
            value, numbers.Integral
        ), "{}: Value = {} must be of type int".format(key, value)
        assert (
            not self.lower
        ) or value >= self.lower, "{}: Value = {} must be >= {}".format(
            key, value, self.lower
        )
        assert (
            not self.upper
        ) or value <= self.upper, "{}: Value = {} must be <= {}".format(
            key, value, self.upper
        )


class IntegerOrNone(Integer):
    def __init__(self, lower: int = None, upper: int = None):
        super().__init__(lower, upper)

    def assert_valid(self, key: str, value):
        if value is not None:
            super().assert_valid(key, value)


class Categorical(CheckType):
    def __init__(self, choices: Tuple[str, ...]):
        self.choices = set(choices)

    def assert_valid(self, key: str, value):
        assert (
            isinstance(value, str) and value in self.choices
        ), "{}: Value = {} must be in {}".format(key, value, self.choices)


class String(CheckType):
    def assert_valid(self, key: str, value):
        assert isinstance(value, str), "{}: Value = {} must be of type str"


class Boolean(CheckType):
    def assert_valid(self, key: str, value):
        assert isinstance(value, bool), "{}: Value = {} must be boolean".format(
            key, value
        )


class Dictionary(CheckType):
    def assert_valid(self, key: str, value):
        assert isinstance(value, dict), "{}: Value = {} must be a dictionary".format(
            key, value
        )


def check_and_merge_defaults(
    options: Dict[str, Any],
    mandatory: Set[str],
    default_options: Dict[str, Any],
    constraints: Optional[Dict[str, CheckType]] = None,
    dict_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    First, check that all keys in mandatory appear in options. Second, create
    result_options by merging ``options`` and ``default_options``, where entries in
    ``options`` have precedence. Finally, if ``constraints`` is given, this is used to
    check validity of values.

    :param options: Input arguments
    :param mandatory: Set of mandatory argument names
    :param default_options: Default values for ``options``
    :param constraints: See above, optional
    :param dict_name: Prefix used in assert messages, optional
    :return: Output arguments
    """
    prefix = "" if dict_name is None else "{}: ".format(dict_name)
    for key in mandatory:
        assert key in options, prefix + "Key '{}' is missing (but is mandatory)".format(
            key
        )
    log_msg = ""
    result_options = {k: v for k, v in options.items() if v is not None}
    for key, value in default_options.items():
        if key not in result_options:
            log_msg += prefix + "Key '{}': Imputing default value {}\n".format(
                key, value
            )
            result_options[key] = value
        # If the argument is a dict, we impute only the missing entries
        if isinstance(value, dict):
            result_dict = result_options[key]
            assert isinstance(
                result_dict, dict
            ), f"Key '{key}': Value must be dictionary, but is {result_dict}"
            for kd, vd in value.items():
                if kd not in result_dict:
                    log_msg += (
                        prefix
                        + "Key '{}' in dict {}: Imputing default value {}\n".format(
                            kd, key, vd
                        )
                    )
                    result_dict[kd] = vd
    if log_msg:
        logger.debug(log_msg.rstrip("\n"))
    # Check constraints
    if constraints:
        for key, value in result_options.items():
            check = constraints.get(key)
            if check:
                check.assert_valid(prefix + "Key '{}'".format(key), value)

    return result_options


def filter_by_key(options: Dict[str, Any], remove_keys: Set[str]) -> Dict[str, Any]:
    """
    Filter options by removing entries whose keys are in ``remove_keys``.
    Used to filter kwargs passed to a constructor, before passing it to
    the superclass constructor.

    :param options: Arguments to be filtered
    :param remove_keys: See above
    :return: Filtered options
    """
    return {k: v for k, v in options.items() if k not in remove_keys}


def assert_no_invalid_options(options: Dict[str, Any], all_keys: Set[str], name: str):
    for k in options:
        assert k in all_keys, "{}: Invalid argument '{}'".format(name, k)
