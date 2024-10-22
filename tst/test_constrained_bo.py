from pathlib import Path
import pytest

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import uniform


# TODO: Turn ``train_constrained_example`` into a blackbox and use the simulator
# backend to run this test rapidly
@pytest.mark.skip("this unit test takes about two minutes and is skipped for now")
@pytest.mark.parametrize(
    "scheduler, searcher, constraint_offset",
    [
        ("fifo", "bayesopt", 1.0),  # ignored constraint
        ("fifo", "bayesopt_constrained", 20.0),  # loose constraint
        ("fifo", "bayesopt_constrained", 1.0),  # strict constraint
        ("fifo", "bayesopt_constrained", -10.0),  # infeasible constraint
    ],
)
def test_constrained_bayesopt(scheduler, searcher, constraint_offset):
    num_workers = 2

    config_space = {
        "x1": uniform(-5, 10),
        "x2": uniform(0, 15),
        "constraint_offset": constraint_offset,  # the lower, the stricter
    }

    trial_backend = LocalBackend(
        entry_point=Path(__file__).parent.parent
        / "examples"
        / "training_scripts"
        / "constrained_hpo"
        / "train_constrained_example.py"
    )

    search_options = {
        "num_init_random": num_workers,
        "constraint_attr": "my_constraint_metric",  # Name of the constraint metric captured
        # by the reporter. If not specified, it is assumed that this is named 'constraint_metric' in the reporter
    }
    stop_criterion = StoppingCriterion(max_wallclock_time=28)

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode="max",
        metric="objective",
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
    )

    tuner.run()
