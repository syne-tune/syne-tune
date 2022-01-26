import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

from benchmarking.blackbox_repository.tabulated_benchmark import \
    BlackboxRepositoryBackend
from benchmarking.benchmark_loop.baselines import methods
from benchmarking.benchmark_loop.benchmark_definitions import \
    benchmark_definitions

from syne_tune.backend.simulator_backend.simulator_callback import \
    SimulatorCallback
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from coolname import generate_slug


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default=generate_slug(2))
    parser.add_argument("--num_seeds", type=int, required=False, default=2)
    parser.add_argument("--method", type=str, required=False)
    parser.add_argument("--benchmark", type=str, required=False)
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    num_seeds = args.num_seeds
    method_names = [args.method] if args.method is not None else list(methods.keys())
    benchmark_names = [args.benchmark] if args.benchmark is not None else list(benchmark_definitions.keys())

    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(logging.WARNING)

    combinations = list(itertools.product(method_names, range(num_seeds), benchmark_names))

    print(combinations)
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}")

        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            time_this_resource_attr=benchmark.time_this_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )

        max_t = max(backend.blackbox.fidelity_values)
        resource_attr = next(iter(backend.blackbox.fidelity_space.keys()))
        scheduler = methods[method](
            config_space=backend.blackbox.configuration_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=seed,
            max_t=max_t,
            resource_attr=resource_attr,
        )

        stop_criterion = StoppingCriterion(max_wallclock_time=benchmark.max_wallclock_time)

        tuner = Tuner(
            backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=600,
            tuner_name=f"{experiment_tag}-{method}-{seed}-{benchmark_name}".replace("_", "-"),
            metadata={
                "seed": seed,
                "algorithm": method,
                "tag": experiment_tag,
                "benchmark": benchmark_name
            }
        )
        tuner.run()