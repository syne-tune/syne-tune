"""
An example showing to launch a tuning of a python function `train_height`.
"""

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import ASHA


def train_height(steps: int, width: float, height: float):
    """
    The function to be tuned, note that import must be in PythonBackend and no global variable are allowed,
    more details on requirements of tuned functions can be found in `PythonBackend`.
    """
    import logging
    from syne_tune import Reporter
    import time
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    reporter = Reporter()
    for step in range(steps):
        dummy_score = (0.1 + width * step / 100) ** (-1) + height * 0.1
        # Feed the score back to Syne Tune.
        reporter(step=step, mean_loss=dummy_score, epoch=step + 1)
        time.sleep(0.1)


if __name__ == '__main__':
    import logging

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }

    scheduler = ASHA(config_space, metric="mean_loss", resource_attr='epoch', max_t=max_steps, mode="min")

    trial_backend = PythonBackend(tune_function=train_height, config_space=config_space)

    stop_criterion = StoppingCriterion(max_wallclock_time=10, min_metric_value={"mean_loss": -6.0})
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )
    tuner.run()
