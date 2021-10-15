from pathlib import Path
import pytest

from sagemaker_tune.backend.local_backend import LocalBackend
from sagemaker_tune.optimizer.schedulers.fifo import FIFOScheduler
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.search_space import uniform
from sagemaker_tune.stopping_criterion import StoppingCriterion


@pytest.mark.skip("this unit test takes about a minute and is skipped for now")
@pytest.mark.parametrize("scheduler, searcher, cost", [
    ('fifo', 'bayesopt', 1),  # ignored cost
    ('fifo', 'bayesopt_cost', 1),  # linear cost
    ('fifo', 'bayesopt_cost', 2),  # quadratic cost
])
def test_cost_aware_bayesopt(scheduler, searcher, cost):
    num_workers = 2

    config_space = {
        "x1": uniform(-5, 10),
        "x2": uniform(0, 15),
        "cost": cost  # cost_value = x2 ** cost
    }

    backend = LocalBackend(
        entry_point=Path(__file__).parent.parent / "examples"
                    / "training_scripts" / "cost_aware_hpo" / "train_cost_aware_example.py")

    search_options = {
        'num_init_random': num_workers,
        'cost_attr': 'elapsed_time',  # Name of the cost metric captured by the reporter (mandatory)
    }
    stop_criterion = StoppingCriterion(max_wallclock_time=18)

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode='max',
        metric='objective',
    )

    tuner = Tuner(
        backend=backend,
        scheduler=myscheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
    )

    tuner.run()
