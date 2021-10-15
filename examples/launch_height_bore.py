import logging
from pathlib import Path

from sagemaker_tune.backend.local_backend import LocalBackend
from sagemaker_tune.optimizer.schedulers.fifo import FIFOScheduler
from sagemaker_tune.optimizer.schedulers.searchers.bore import Bore
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.search_space import randint
from sagemaker_tune.stopping_criterion import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    max_trials = 100
    n_workers = 1  # BORE only works for the non-parallel setting so far

    config_space = {
        'steps': 100,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = Path(__file__).parent / "training_scripts" / "height_example" / "train_height.py"
    mode = "min"
    metric = "mean_loss"

    backend = LocalBackend(entry_point=str(entry_point))

    bore = Bore(config_space=config_space, metric=metric, mode=mode)
    scheduler = FIFOScheduler(
        config_space,
        searcher=bore,
        mode=mode,
        metric=metric)

    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=120),
        n_workers=n_workers,
    )

    tuner.run()
