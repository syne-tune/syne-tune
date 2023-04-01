# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import DyHPO
from syne_tune.results_callback import FinalResultsComposer, StoreResultsCallback
from syne_tune import Tuner, StoppingCriterion


# We would like to extract some information from the scheduler at the end of
# the experiment. To this end, we implement a class for extracting this
# information
class DyHPOFinalResults(FinalResultsComposer):
    def __call__(self, tuner: Tuner) -> Optional[Dict[str, Any]]:
        scheduler = tuner.scheduler
        assert isinstance(scheduler, DyHPO)  # sanity check
        # :class:`~syne_tune.optimizer.schedulers.searchers.dyhpo.hyperband_dyhpo.DyHPORungSystem`
        # collects statistics about how often several types of decisions were made in
        # ``on_task_schedule``. We would like to store this information as final
        # results
        return {
            "summary_schedule_records": scheduler.terminator._rung_systems[
                0
            ].summary_schedule_records()
        }


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

    # We use the DyHPO scheduler, since it records some informations for the final
    # results
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

    # Final results are stored by the
    # :class:`~syne_tune.results_callback.StoreResultsCallback` which also stores
    # time-stamped results. We need to pass an object to compose the final results
    final_results_composer = DyHPOFinalResults()
    callbacks = [StoreResultsCallback(final_results_composer=final_results_composer)]
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=4,  # how many trials are evaluated in parallel
        callbacks=callbacks,
    )
    tuner.run()

    # Let us have a look what was written as final results
    experiment_results = load_experiment(tuner.name)
    print(f"\nFinal results written:\n{experiment_results.final_results}")
