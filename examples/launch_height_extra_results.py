from typing import Any, Optional
from pathlib import Path
import logging

from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.legacy_baselines import DyHPO
from syne_tune.optimizer.schedulers.searchers.legacy_dyhpo.hyperband_dyhpo import (
    DyHPORungSystem,
)
from syne_tune.results_callback import ExtraResultsComposer, StoreResultsCallback
from syne_tune import Tuner, StoppingCriterion


# We would like to extract some extra information from the scheduler during the
# experiment. To this end, we implement a class for extracting this information
class DyHPOExtraResults(ExtraResultsComposer):
    def __call__(self, tuner: Tuner) -> Optional[dict[str, Any]]:
        scheduler = tuner.scheduler
        assert isinstance(scheduler, DyHPO)  # sanity check
        # :class:`~syne_tune.optimizer.schedulers.searchers.legacy_dyhpo.hyperband_dyhpo.DyHPORungSystem`
        # collects statistics about how often several types of decisions were made in
        # ``on_task_schedule``
        return scheduler.terminator._rung_systems[0].summary_schedule_records()

    def keys(self) -> list[str]:
        return DyHPORungSystem.summary_schedule_keys()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_epochs = 100
    n_workers = 4
    # Hyperparameter configuration space
    config_space = {
        "width": randint(1, 20),
        "height": randint(1, 20),
        "epochs": 100,
    }

    # We use the DyHPO scheduler, since it records some interesting extra
    # informations
    scheduler = DyHPO(
        config_space,
        metric="mean_loss",
        resource_attr="epoch",
        max_resource_attr="epochs",
        search_options={"debug_log": False},
        grace_period=2,
    )
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height_simple.py"
    )

    # Extra results are stored by the
    # :class:`~syne_tune.results_callback.StoreResultsCallback`. In fact, they
    # are appended to the default time-stamped results whenever a report is
    # received.
    extra_results_composer = DyHPOExtraResults()
    callbacks = [StoreResultsCallback(extra_results_composer=extra_results_composer)]
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=4,  # how many trials are evaluated in parallel
        callbacks=callbacks,
    )
    tuner.run()

    # Let us have a look what was written. Here, we just look at the information
    # at the end of the experiment
    results_df = load_experiment(tuner.name).results
    final_pos = results_df.loc[:, ST_TUNER_TIME].argmax()
    final_row = dict(results_df.loc[final_pos])
    extra_results_at_end = {
        name: final_row[name] for name in extra_results_composer.keys()
    }
    print(f"\nExtra results at end of experiment:\n{extra_results_at_end}")
