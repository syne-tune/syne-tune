import logging
from typing import List, Optional, Dict, Any

from syne_tune.optimizer.schedulers.searchers.legacy_gp_fifo_searcher import (
    GPFIFOSearcher,
)
from syne_tune.optimizer.schedulers.searchers.legacy_gp_searcher_factory import (
    cost_aware_gp_fifo_searcher_defaults,
    cost_aware_coarse_gp_fifo_searcher_factory,
    cost_aware_fine_gp_fifo_searcher_factory,
)
from syne_tune.optimizer.schedulers.searchers.legacy_gp_searcher_utils import (
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    ModelStateTransformer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)

logger = logging.getLogger(__name__)


class MultiModelGPFIFOSearcher(GPFIFOSearcher):
    """
    Superclass for multi-model extensions of
    :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`. We first
    call
    :meth:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher._create_internal`
    passing factory and ``skip_optimization`` predicate for the ``INTERNAL_METRIC_NAME``
    model, then replace the state transformer by a multi-model one.
    """

    def _call_create_internal(self, kwargs_int):
        output_estimator = kwargs_int.pop("output_estimator")
        output_skip_optimization = kwargs_int.pop("output_skip_optimization")
        kwargs_int["estimator"] = output_estimator[INTERNAL_METRIC_NAME]
        kwargs_int["skip_optimization"] = output_skip_optimization[INTERNAL_METRIC_NAME]
        self._create_internal(**kwargs_int)
        # Replace ``state_transformer``
        init_state = self.state_transformer.state
        self.state_transformer = ModelStateTransformer(
            estimator=output_estimator,
            init_state=init_state,
            skip_optimization=output_skip_optimization,
        )


class CostAwareGPFIFOSearcher(MultiModelGPFIFOSearcher):
    """
    Gaussian process-based cost-aware hyperparameter optimization (to be used
    with :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`). The searcher
    requires a cost metric, which is given by ``cost_attr``.

    Implements two different variants. If ``resource_attr`` is given, cost values
    are read from each report and cost is modeled as :math:`c(x, r)`, the cost
    model being given by ``kwargs["cost_model"]``.

    If ``resource_attr`` is not given, cost values are read only at the end (just
    like the primary metric) and cost is modeled as :math:`c(x)`, using a
    default GP surrogate model.

    Note: The presence or absence of ``resource_attr`` decides on which variant
    is used here. If ``resource_attr`` is given, ``cost_model`` must be given
    as well.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`:

    :param cost_attr: Mandatory. Name of cost attribute in data obtained
        from reporter (e.g., elapsed training time). Depending on whether
        ``resource_attr`` is given, cost values are read from each report or
        only at the end.
    :type cost_attr: str
    :param resource_attr: Name of resource attribute in reports, optional.
        If this is given, cost values are read from each report and cost is
        modeled as :math:`c(x, r)`, the cost model being given by ``cost_model``.
        If not given, cost values are read only at the end (just like the
        primary metric) and cost is modeled as :math:`c(x)`, using a default
        GP surrogate model.
    :type resource_attr: str, optional
    :param cost_model: Needed if ``resource_attr`` is given, model for
        :math:`c(x, r)`. Ignored if ``resource_attr`` is not given, since
        :math:`c(x)` is represented by a default GP surrogate model.
    :type cost_model:
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.cost_model.CostModel`, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        assert kwargs.get("cost_attr") is not None, (
            "This searcher needs a cost attribute. Please specify its "
            + "name in search_options['cost_attr']"
        )
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs,
            *cost_aware_gp_fifo_searcher_defaults(kwargs),
            dict_name="search_options",
        )
        # If ``resource_attr`` is specified, we do fine-grained, otherwise
        # coarse-grained
        if kwargs.get("resource_attr") is not None:
            logger.info(
                "Fine-grained: Modelling cost values c(x, r) "
                + "obtained at every resource level r"
            )
            kwargs_int = cost_aware_fine_gp_fifo_searcher_factory(**_kwargs)
        else:
            logger.info(
                "Coarse-grained: Modelling cost values c(x) "
                + "obtained together with metric values"
            )
            kwargs_int = cost_aware_coarse_gp_fifo_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        output_skip_optimization = state["skip_optimization"]
        output_estimator = self.state_transformer.estimator
        # Call internal constructor
        new_searcher = CostAwareGPFIFOSearcher(
            **self._new_searcher_kwargs_for_clone(),
            output_estimator=output_estimator,
            init_state=init_state,
            output_skip_optimization=output_skip_optimization,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
