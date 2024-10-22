"""
Example for how to tune one of the benchmarks.
"""
import logging

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune import Tuner, StoppingCriterion

from benchmarking.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    # We pick the MLP on FashionMNIST benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    random_seed = 31415927
    n_workers = 4
    benchmark = mlp_fashionmnist_benchmark()

    # If you don't like the default config_space, change it here. But let
    # us use the default
    config_space = benchmark.config_space

    # Local backend
    trial_backend = LocalBackend(entry_point=str(benchmark.script))

    # GP-based Bayesian optimization searcher. Many options can be specified
    # via ``search_options``, but let's use the defaults
    searcher = "bayesopt"
    search_options = {"num_init_random": n_workers + 2}
    # Hyperband (or successive halving) scheduler of the stopping type.
    # Together with 'bayesopt', this selects the MOBSTER algorithm.
    # If you don't like the defaults suggested, just change them:
    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        max_resource_attr=benchmark.max_resource_attr,
        resource_attr=benchmark.resource_attr,
        mode=benchmark.mode,
        metric=benchmark.metric,
        random_seed=random_seed,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=120)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
