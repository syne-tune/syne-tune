from sagemaker_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    GPFIFOSearcher
from sagemaker_tune.optimizer.schedulers.searchers.constrained_gp_fifo_searcher \
    import ConstrainedGPFIFOSearcher
from sagemaker_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher \
    import CostAwareGPFIFOSearcher
from sagemaker_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher \
    import GPMultiFidelitySearcher
from sagemaker_tune.optimizer.schedulers.searchers.gpiss_multifidelity_searcher \
    import GPISSMultiFidelitySearcher
from sagemaker_tune.optimizer.schedulers.searchers.cost_aware_gp_multifidelity_searcher \
    import CostAwareGPMultiFidelitySearcher
from sagemaker_tune.optimizer.schedulers.searchers.searcher import \
    RandomSearcher

__all__ = ['searcher_factory']


_OUR_MULTIFIDELITY_SCHEDULERS = {
    'hyperband_stopping', 'hyperband_promotion', 'hyperband_cost_promotion', 'hyperband_pasha'}

_OUR_SCHEDULERS = _OUR_MULTIFIDELITY_SCHEDULERS | {'fifo'}

SEARCHER_CONFIGS = dict(
    random=dict(
        searcher_cls=RandomSearcher,
    ),
    bayesopt=dict(
        # Gaussian process based Bayesian optimization
        # The searchers and their kwargs differ depending on the scheduler
        # type (fifo, hyperband_*)
        searcher_cls=lambda scheduler: \
            GPFIFOSearcher if scheduler == 'fifo' \
                else GPMultiFidelitySearcher,
        supported_schedulers=_OUR_SCHEDULERS,
    ),
    bayesopt_issm=dict(
        # Bayesian optimization with GP-ISS model
        searcher_cls=GPISSMultiFidelitySearcher,
        supported_schedulers=_OUR_MULTIFIDELITY_SCHEDULERS,
    ),
    bayesopt_constrained=dict(
        # Gaussian process based Constrained Bayesian optimization
        searcher_cls=ConstrainedGPFIFOSearcher,
        supported_schedulers={'fifo'},
    ),
    bayesopt_cost=dict(
        # Gaussian process based Cost-Aware Bayesian optimization
        # The searchers and their kwargs differ depending on the scheduler
        # type (fifo, hyperband_*)
        searcher_cls=lambda scheduler: \
            CostAwareGPFIFOSearcher if scheduler == 'fifo' \
                else CostAwareGPMultiFidelitySearcher,
        supported_schedulers=_OUR_SCHEDULERS,
    ),
)


def searcher_factory(searcher_name, **kwargs):
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see :class:`FIFOScheduler`), which provides most of the required
    kwargs.

    """
    if searcher_name in SEARCHER_CONFIGS:
        searcher_config = SEARCHER_CONFIGS[searcher_name]
        searcher_cls = searcher_config['searcher_cls']
        scheduler = kwargs.get('scheduler')

        # Check if searcher_cls is a lambda - evaluate then
        if isinstance(searcher_cls, type(lambda: 0)):
            searcher_cls = searcher_cls(scheduler)

        if 'supported_schedulers' in searcher_config:
            supported_schedulers = searcher_config['supported_schedulers']
            assert scheduler is not None, \
                "Scheduler must set search_options['scheduler']"
            assert scheduler in supported_schedulers, \
                f"Searcher '{searcher_name}' only works with schedulers {supported_schedulers} (not with '{scheduler}')"

        searcher = searcher_cls(**kwargs)
        return searcher
    else:
        raise AssertionError(f"searcher '{searcher_name}' is not supported")
