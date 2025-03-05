"""
Example for running legacy_constrained Bayesian optimization on a toy example
"""
import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import LegacyFIFOScheduler
from syne_tune.config_space import uniform
from syne_tune import StoppingCriterion, Tuner


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 2

    config_space = {
        "x1": uniform(-5, 10),
        "x2": uniform(0, 15),
        "constraint_offset": 1.0,  # the lower, the stricter
    }

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "constrained_hpo"
        / "train_constrained_example.py"
    )
    mode = "max"
    metric = "objective"
    constraint_attr = "my_constraint_metric"

    # Local backend
    trial_backend = LocalBackend(entry_point=entry_point)

    # Bayesian legacy_constrained optimization:
    #   :math:`max_x f(x),   \mathrm{s.t.} c(x) <= 0`
    # Here, ``metric`` represents :math:`f(x)`, ``constraint_attr`` represents
    # :math:`c(x)`.
    search_options = {
        "num_init_random": n_workers,
        "constraint_attr": constraint_attr,
    }
    scheduler = LegacyFIFOScheduler(
        config_space,
        searcher="bayesopt_constrained",
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=20)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
