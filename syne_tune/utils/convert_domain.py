# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.
from typing import Dict, Any, Union, Optional, List
from numbers import Real
import numpy as np
import logging

from syne_tune.config_space import (
    Categorical,
    Float,
    Integer,
    Domain,
    finrange,
    logfinrange,
    ordinal,
    logordinal,
    is_log_space,
    loguniform,
    lograndint,
)
from syne_tune.util import is_integer

logger = logging.getLogger(__name__)


def fit_to_regular_grid(x: np.ndarray) -> Dict[str, float]:
    r"""
    Computes the least squares fit of :math:`a * j + b` to ``x[j]``, where
    :math:`j = 0,\dots, n-1`. Returns the LS estimate of ``a``, ``b``, and the
    coefficient of variation :math:`R^2`.

    :param x: Strictly increasing sequence
    :return: See above
    """
    n = x.size
    assert n > 1
    # Linear transformation of equations gives:
    # :math:`x[j+1] - x[j] \approx a` for :math:`j = 0,\dots, n-2`,
    # so the least squares solution for ``a`` is the mean of the difference sequence,
    # and the least squares solution for ``b`` is obtained by plugging this solution in:
    # :math:`b \approx x[j] - \hat{a} * j` for :math:`j = 0,\dots, n-1`.
    ls_a = (x[-1] - x[0]) / (n - 1)
    u = x - ls_a * np.arange(0, n)
    ls_b = np.mean(u)
    r_squared = 1 - np.var(u) / np.var(x)
    return {
        "r2": r_squared,
        "a": ls_a,
        "b": ls_b,
    }


POSITIVE_EPS = 1e-20


R2_THRESHOLD = 0.995


def _is_choice_domain(domain: Domain) -> bool:
    return isinstance(domain, Categorical)


def convert_choice_domain(domain: Categorical, name: Optional[str] = None) -> Domain:
    """
    If the choice domain ``domain`` has more than 2 numerical values, it is
    converted to :func:`~syne_tune.config_space.finrange`,
    :func:`~syne_tune.config_space.logfinrange`,
    :func:`~syne_tune.config_space.ordinal`, or
    :func:`~syne_tune.config_space.logordinal`. Otherwise, ``domain`` is
    returned as is.

    The idea is to compute the least squares fit :math:`a * j + b` to ``x[j]``,
    where ``x`` are the sorted values or their logs (if all values are positive).
    If this fit is very close (judged by coefficient of variation :math:`R^2`), we
    use the equispaced types ``finrange`` or ``logfinrange``, otherwise we use
    ``ordinal`` or ``logordinal``.
    """
    num_values = len(domain)
    if num_values <= 2:
        return domain
    if not isinstance(domain.categories[0], Real):
        # Note: All entries of ``domain.categories`` have the same type
        return domain
    values_are_int = is_integer(domain.categories)
    sorted_values = sorted(domain.categories)
    x = np.array(sorted_values)
    best_fit = fit_to_regular_grid(x)
    best_is_log = False
    if sorted_values[0] >= POSITIVE_EPS:
        # All entries are positive. Try least squares fit in log domain
        log_fit = fit_to_regular_grid(np.log(x))
        if log_fit["r2"] > best_fit["r2"]:
            # Better fit in log domain
            best_is_log = True
            best_fit = log_fit
    if best_fit["r2"] >= R2_THRESHOLD:
        # Error of least squares fit in normal or log domain small enough:
        # Use ``finrange`` or ``logfinrange``
        lower = best_fit["b"]
        upper = lower + best_fit["a"] * (num_values - 1)
        if best_is_log:
            result = logfinrange(
                lower=np.exp(lower),
                upper=np.exp(upper),
                size=num_values,
                cast_int=values_are_int,
            )
        else:
            result = finrange(
                lower=lower, upper=upper, size=num_values, cast_int=values_are_int
            )
    else:
        # Least squares fit not good enough: Use ``ordinal`` or ``logordinal``
        result = (
            logordinal(sorted_values)
            if best_is_log
            else ordinal(sorted_values, kind="nn")
        )
    if name is not None:
        logger.info(
            f"{name}: is_log = {best_is_log}, R2 = {best_fit['r2']}:\n"
            f"{domain} -> {result}"
        )
    return result


def _is_float_or_int_domain(domain: Domain) -> bool:
    return isinstance(domain, Float) or isinstance(domain, Integer)


UPPER_LOWER_RATIO_THRESHOLD = 100


def convert_linear_to_log_domain(
    domain: Union[Float, Integer], name: Optional[str] = None
) -> Domain:
    if is_log_space(domain) or domain.lower < POSITIVE_EPS:
        return domain
    if domain.upper <= domain.lower * UPPER_LOWER_RATIO_THRESHOLD:
        return domain
    result = (
        loguniform(domain.lower, domain.upper)
        if isinstance(domain, Float)
        else lograndint(domain.lower, domain.upper)
    )
    if name is not None:
        logger.info(f"{name}: {domain} -> {result}")
    return result


def convert_domain(domain: Domain, name: Optional[str] = None) -> Domain:
    """
    If one of the following rules apply, ``domain`` is converted and returned,
    otherwise it is returned as is.

    * ``domain`` is categorical, its values are numerical. This is converted to
      :func:`~syne_tune.config_space.finrange`,
      :func:`~syne_tune.config_space.logfinrange`,
      :func:`~syne_tune.config_space.ordinal`, or
      :func:`~syne_tune.config_space.logordinal`. We fit the values or their
      logs to the closest regular grid, converting to ``(log)finrange`` if the
      least squares fit to the grid is good enough, otherwise to
      ``(log)ordinal``, where ``ordinal`` is with ``kind="nn"``. Note that the
      conversion to ``(log)finrange`` may result in slightly different values.
    * ``domain`` is ``float` or ``int``. This is converted to the same type, but
      in log scale, if the current scale is linear, ``lower`` is positive, and
      the ratio ``upper / lower`` is larger than :const:`UPPER_LOWER_RATIO_THRESHOLD`.

    :param domain: Original domain
    :return: Streamlined domain
    """
    if _is_choice_domain(domain):
        return convert_choice_domain(domain, name)
    elif _is_float_or_int_domain(domain):
        return convert_linear_to_log_domain(domain, name)
    else:
        return domain


def streamline_config_space(
    config_space: Dict[str, Any],
    exclude_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Given a configuration space ``config_space``, this function returns a new
    configuration space where some domains may have been replaced by approximately
    equivalent ones, which are however better suited for Bayesian optimization. Entries
    with key in ``exclude_names`` are not replaced.

    See :func:`convert_domain` for what replacement rules may be applied.

    :param config_space: Original configuration space
    :param exclude_names: Do not convert entries with these keys
    :param verbose: Log output for replaced domains? Defaults to ``False``
    :return: Streamlined configuration space
    """
    if exclude_names is None:
        exclude_names = []

    def _convert(name, domain):
        if name not in exclude_names and isinstance(domain, Domain):
            return convert_domain(domain, name=name if verbose else None)
        else:
            return domain

    return {name: _convert(name, domain) for name, domain in config_space.items()}
