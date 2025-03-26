"""
Example for cost-aware promotion-based Hyperband
"""
import logging

from benchmarking.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)
from benchmarking.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import (
    ELAPSED_TIME_ATTR,
)
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import LegacyHyperbandScheduler
from syne_tune import Tuner, StoppingCriterion


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # logging.getLogger().setLevel(logging.INFO)

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

    # Cost-aware variant of ASHA, using a random searcher
    scheduler = LegacyHyperbandScheduler(
        config_space,
        searcher="random",
        max_resource_attr=benchmark.max_resource_attr,
        resource_attr=benchmark.resource_attr,
        mode=benchmark.mode,
        metric=benchmark.metric,
        type="cost_promotion",
        cost_attr=ELAPSED_TIME_ATTR,
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
