from pathlib import Path

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.legacy_baselines import ASHA

# Hyperparameter configuration space
config_space = {
    "width": randint(1, 20),
    "height": randint(1, 20),
    "epochs": 100,
}
# Scheduler (i.e., HPO algorithm)
scheduler = ASHA(
    config_space,
    metric="mean_loss",
    resource_attr="epoch",
    max_resource_attr="epochs",
    search_options={"debug_log": False},
)

entry_point = str(
    Path(__file__).parent
    / "training_scripts"
    / "height_example"
    / "train_height_simple.py"
)
tuner = Tuner(
    trial_backend=LocalBackend(entry_point=entry_point),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_wallclock_time=30),
    n_workers=4,  # how many trials are evaluated in parallel
)
tuner.run()
