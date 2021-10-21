import logging
from pathlib import Path

from sagemaker_tune.backend.local_backend import LocalBackend
from sagemaker_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.search_space import randint
from sagemaker_tune.stopping_criterion import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = Path(__file__).parent / "training_scripts" / "height_example" / "train_height.py"
    mode = "min"
    metric = "mean_loss"

    # Local back-end
    backend = LocalBackend(entry_point=str(entry_point))

    searcher = 'random'
    hyperband_type = 'pasha'
    search_options = {'num_init_random': n_workers + 2}
    ranking_criterion = 'soft_ranking'
    
    # epsilon is used for soft_ranking
    epsilon = 5.0
    # epsilon_scaling is used for soft_ranking_std, soft_ranking_median_dst, soft_ranking_mean_dst
    epsilon_scaling = 0.5

    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        type=hyperband_type,
        search_options=search_options,
        max_t=max_steps,
        resource_attr='epoch',
        mode=mode,
        metric=metric,
        random_seed=random_seed,
        ranking_criterion=ranking_criterion,
        epsilon=epsilon,
        epsilon_scaling=epsilon_scaling
        )

    stop_criterion = StoppingCriterion(max_num_trials_started=30)
    
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0.01
    )

    tuner.run()
