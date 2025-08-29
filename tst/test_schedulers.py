import pytest

from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving

from tst.util_test import run_experiment_with_height


def make_asha_scheduler(searcher):
    def maker(config_space, metric, do_minimize, random_seed, resource_attr):

        myscheduler = AsynchronousSuccessiveHalving(
            config_space,
            searcher=searcher,
            resource_attr=resource_attr,
            random_seed=random_seed,
            do_minimize=do_minimize,
            metric=metric,
        )
        return myscheduler

    return maker


asha_variants = ["cqr", "random_search", "kde", "bore"]


@pytest.mark.skip()
@pytest.mark.timeout(10)
@pytest.mark.parametrize("searcher", asha_variants)
def test_asha_schedulers_local(searcher):
    run_experiment_with_height(
        make_scheduler=make_asha_scheduler(searcher),
        simulated=False,
    )


def make_single_objective_scheduler(searcher):
    def maker(config_space, metric, do_minimize, random_seed, resource_attr):

        myscheduler = SingleObjectiveScheduler(
            config_space,
            searcher=searcher,
            random_seed=random_seed,
            do_minimize=do_minimize,
            metric=metric,
        )
        return myscheduler

    return maker


asha_variants = ["cqr", "random_search", "kde", "regularized_evolution", "bore"]


@pytest.mark.skip()
@pytest.mark.timeout(10)
@pytest.mark.parametrize("searcher", asha_variants)
def test_single_objective_schedulers_local(searcher):
    run_experiment_with_height(
        make_scheduler=make_asha_scheduler(searcher),
        simulated=False,
    )
