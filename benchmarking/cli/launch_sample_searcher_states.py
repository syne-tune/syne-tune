"""
This script launches an experiment for the purpose of sampling searcher
states, which can then be used in unit tests.
"""
import logging

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune import Tuner

from benchmarking.definitions.definition_mlp_on_fashion_mnist import \
    mlp_fashionmnist_benchmark, mlp_fashionmnist_default_params
from benchmarking.cli.searcher_state_callback import StoreSearcherStatesCallback


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    # We pick the MLP on FashionMNIST benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    random_seed = 31415927
    n_workers = 4
    #generate_data_for = 'test_expdecay_model'
    generate_data_for = 'test_iss_model'

    default_params = mlp_fashionmnist_default_params()
    benchmark = mlp_fashionmnist_benchmark(default_params)
    mode = benchmark['mode']
    metric = benchmark['metric']
    config_space = benchmark['config_space']

    # Local back-end
    trial_backend = LocalBackend(entry_point=benchmark['script'])

    # GP-based Bayesian optimization searcher
    searcher = 'bayesopt'
    if generate_data_for == 'test_expdecay_model':
        search_options = {
            'num_init_random': 6,  # Good value for 4 workers
            'model': 'gp_multitask',
            'gp_resource_kernel': 'freeze-thaw',
        }
    else:
        assert generate_data_for == 'test_iss_model'
        search_options = {
            'num_init_random': 6,  # Good value for 4 workers
            'model': 'gp_issm',
            'issm_gamma_one': False,
        }
    # Hyperband (or successive halving) scheduler of the stopping type.
    # Together with 'bayesopt', this selects the MOBSTER algorithm.
    # If you don't like the defaults suggested, just change them:
    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        max_t=default_params['max_resource_level'],
        grace_period=default_params['grace_period'],
        reduction_factor=default_params['reduction_factor'],
        resource_attr=benchmark['resource_attr'],
        mode=mode,
        metric=metric,
        random_seed=random_seed,
        searcher_data='all',  # We need densely sampled data
    )

    callback = StoreSearcherStatesCallback()
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=lambda status: status.wallclock_time > 600,
        n_workers=n_workers,
        callbacks=[callback]
    )

    tuner.run()

    print(f"Number of searcher states logged: {len(callback.states)}")
    print("Here is code for them:")
    for pos in range(len(callback.states)):
        print(f"\nSearcher state {pos}")
        print(callback.searcher_state_as_code(pos, add_info=True))
