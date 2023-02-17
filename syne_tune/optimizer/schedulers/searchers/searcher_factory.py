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

from syne_tune.try_import import (
    try_import_gpsearchers_message,
    try_import_kde_message,
    try_import_bore_message,
    try_import_botorch_message,
)
from syne_tune.optimizer.schedulers.searchers import (
    BaseSearcher,
    RandomSearcher,
    GridSearcher,
)

__all__ = ["searcher_factory"]

logger = logging.getLogger(__name__)


SUPPORTED_SEARCHERS_FIFO = {
    "random",
    "grid",
    "kde",
    "bore",
    "botorch",
    "bayesopt",
    "bayesopt_constrained",
    "bayesopt_cost",
}


SUPPORTED_SEARCHERS_HYPERBAND = {
    "random",
    "grid",
    "kde",
    "bore",
    "bayesopt",
    "bayesopt_cost",
    "hypertune",
    "dyhpo",
}


_OUR_MULTIFIDELITY_SCHEDULERS = {
    "hyperband_stopping",
    "hyperband_promotion",
    "hyperband_cost_promotion",
    "hyperband_pasha",
    "hyperband_synchronous",
}


def searcher_factory(searcher_name: str, **kwargs) -> BaseSearcher:
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`),
    which provides most of the required ``kwargs``.

    :param searcher_name: Value of ``searcher`` argument to scheduler (see
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`)
    :param kwargs: Argument to
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher` constructor
    :return: New searcher object
    """
    supported_schedulers = None
    scheduler = kwargs.get("scheduler")
    model = kwargs.get("model", "gp_multitask")
    if searcher_name == "random":
        searcher_cls = RandomSearcher
    elif searcher_name == "grid":
        searcher_cls = GridSearcher
    elif searcher_name == "kde":
        try:
            from syne_tune.optimizer.schedulers.searchers.kde import (
                KernelDensityEstimator,
                MultiFidelityKernelDensityEstimator,
            )
        except ImportError:
            logger.info(try_import_kde_message())
            raise

        if scheduler == "fifo":
            searcher_cls = KernelDensityEstimator
        else:
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = MultiFidelityKernelDensityEstimator
    elif searcher_name == "bore":
        try:
            from syne_tune.optimizer.schedulers.searchers.bore import (
                Bore,
                MultiFidelityBore,
            )
        except ImportError:
            logger.info(try_import_bore_message())
            raise

        if scheduler == "fifo":
            searcher_cls = Bore
        else:
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = MultiFidelityBore
    elif searcher_name == "botorch":
        try:
            from syne_tune.optimizer.schedulers.searchers.botorch import (
                BoTorchSearcher,
            )
        except ImportError:
            logger.info(try_import_botorch_message())
            raise

        searcher_cls = BoTorchSearcher
        supported_schedulers = {"fifo"}
    else:
        gp_searchers = {
            "bayesopt",
            "bayesopt_constrained",
            "bayesopt_cost",
            "hypertune",
            "dyhpo",
        }
        assert (
            searcher_name in gp_searchers
        ), f"searcher '{searcher_name}' is not supported"
        try:
            from syne_tune.optimizer.schedulers.searchers import (
                GPFIFOSearcher,
                GPMultiFidelitySearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.constrained import (
                ConstrainedGPFIFOSearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.cost_aware import (
                CostAwareGPFIFOSearcher,
                CostAwareGPMultiFidelitySearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.hypertune import (
                HyperTuneSearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.dyhpo import (
                DynamicHPOSearcher,
            )
        except ImportError:
            logger.info(try_import_gpsearchers_message())
            raise

        if searcher_name == "bayesopt":
            if scheduler == "fifo":
                searcher_cls = GPFIFOSearcher
            else:
                supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
                if (
                    model == "gp_multitask"
                    and kwargs.get("gp_resource_kernel") == "freeze-thaw"
                ):
                    logger.warning(
                        "You are combining model = gp_multitask with "
                        "gp_resource_kernel = freeze-thaw. This is mainly "
                        "for debug purposes. The same surrogate model is "
                        "obtained with model = gp_expdecay, but computations "
                        "are faster then."
                    )
                searcher_cls = GPMultiFidelitySearcher
        elif searcher_name == "hypertune":
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = HyperTuneSearcher
        elif searcher_name == "bayesopt_constrained":
            supported_schedulers = {"fifo"}
            searcher_cls = ConstrainedGPFIFOSearcher
        elif searcher_name == "dyhpo":
            supported_schedulers = {"hyperband_dyhpo"}
            searcher_cls = DynamicHPOSearcher
        else:  # bayesopt_cost
            if scheduler == "fifo":
                searcher_cls = CostAwareGPFIFOSearcher
            else:
                supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
                searcher_cls = CostAwareGPMultiFidelitySearcher

    if supported_schedulers is not None:
        assert scheduler is not None, "Scheduler must set search_options['scheduler']"
        assert scheduler in supported_schedulers, (
            f"Searcher '{searcher_name}' only works with schedulers "
            + f"{supported_schedulers} (not with '{scheduler}')"
        )
    searcher = searcher_cls(**kwargs)
    return searcher
