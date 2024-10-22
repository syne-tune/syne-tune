from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


RungSystemsPerBracket = List[List[Tuple[int, int]]]


class SynchronousHyperbandRungSystem:
    """
    Collects factory methods for ``RungSystemsPerBracket`` rung systems to be
    used in :class:`SynchronousHyperbandBracketManager`.

    """

    @staticmethod
    def geometric(
        min_resource: int,
        max_resource: int,
        reduction_factor: float,
        num_brackets: Optional[int] = None,
    ) -> RungSystemsPerBracket:
        """
        This is the geometric progression setup from the original papers on
        successive halving and Hyperband.

        If ``smax = ceil(log(max_resource / min_resource) /
        log(reduction_factor))``, there can be at most ``s_max + 1`` brackets.
        Here, bracket s has ``r_num = s_max - s + 1`` rungs, and the size of
        rung r in bracket s is
            ``n(r,s) = ceil( (s_max + 1) / r_num) *
            power(reduction_factor, r_num - r - 1)``

        :param min_resource: Smallest resource level (positive int)
        :param max_resource: Largest resource level (positive int)
        :param reduction_factor: Approximate ratio between successive rung levels
        :param num_brackets: Number of brackets. If not given, the maximum number
            of brackets is used. Pass 1 for successive halving
        :return: Rung system
        """
        SynchronousHyperbandRungSystem._assert_positive_int(
            min_resource, "min_resource"
        )
        SynchronousHyperbandRungSystem._assert_positive_int(
            max_resource, "max_resource"
        )
        assert min_resource < max_resource
        assert (
            reduction_factor >= 2
        ), f"reduction_factor = {reduction_factor} must be >= 2"
        s_max = 0
        while min_resource * np.power(reduction_factor, s_max) < max_resource:
            s_max += 1
        if num_brackets is not None:
            SynchronousHyperbandRungSystem._assert_positive_int(
                num_brackets, "num_brackets"
            )
        else:
            num_brackets = s_max + 1  # Max number of brackets
        msg_prefix = (
            f"min_resource = {min_resource}, max_resource = "
            + f"{max_resource}, reduction_factor = {reduction_factor}"
        )
        if s_max <= 0:
            logger.warning(
                msg_prefix + ": supports only one bracket with a single rung level of "
                "size 1. Is that really what you want?"
            )
            return [[(1, max_resource)]]
        if num_brackets > s_max + 1:
            logger.warning(
                msg_prefix
                + f": does not support num_brackets = {num_brackets}, but at "
                f"most {s_max + 1}. I am switching to the latter one."
            )
            num_brackets = s_max + 1
        rung_systems = []
        for bracket in range(num_brackets):
            rungs = []
            r_num_m1 = s_max - bracket
            pre_fact = (s_max + 1) / (r_num_m1 + 1)
            for rung in range(r_num_m1):
                resource = int(
                    round(min_resource * np.power(reduction_factor, rung + bracket))
                )
                rsize = int(
                    np.ceil(pre_fact * np.power(reduction_factor, r_num_m1 - rung))
                )
                rungs.append((rsize, resource))
            rungs.append((int(np.ceil(pre_fact)), max_resource))
            rung_systems.append(rungs)
        return rung_systems

    @staticmethod
    def _assert_positive_int(x: int, name: str):
        assert round(x) == x and x >= 1, f"{name} = {x} must be a positive integer"
