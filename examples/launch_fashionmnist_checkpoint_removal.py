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
"""
Example for speculative checkpoint removal with asynchronous multi-fidelity
"""
from typing import Optional, Dict, Any, List
import logging

from syne_tune.backend import LocalBackend
from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCommon,
)
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import MOBSTER
from syne_tune.results_callback import ExtraResultsComposer, StoreResultsCallback
from syne_tune import Tuner, StoppingCriterion

from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)


# This is used to monitor what the checkpoint removal mechanism is doing, and
# writing out results. This is optional, the mechanism works without this.
class CPRemovalExtraResults(ExtraResultsComposer):
    def __call__(self, tuner: Tuner) -> Optional[Dict[str, Any]]:
        result = None
        callback = tuner.callbacks[-1]
        if isinstance(callback, HyperbandRemoveCheckpointsCommon):
            result = callback.extra_results()
        return result

    def keys(self) -> List[str]:
        return HyperbandRemoveCheckpointsCommon.extra_results_keys()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    random_seed = 31415927
    n_workers = 4
    max_num_checkpoints = 10
    # This time may be too short to see positive effects:
    max_wallclock_time = 1800
    # Monitor how checkpoint removal is doing over time, appending this
    # information to results.csv.zip?
    monitor_cp_removal_in_results = True

    # We pick the MLP on FashionMNIST benchmark
    benchmark = mlp_fashionmnist_benchmark()

    # Local backend
    # By setting ``delete_checkpoints=True``, we ask for checkpoints to be removed
    # once a trial cannot be resumed anymore
    trial_backend = LocalBackend(
        entry_point=str(benchmark.script),
        delete_checkpoints=True,
    )

    # MOBSTER (model-based ASHA) with promotion scheduling (pause and resume).
    # Checkpoints are written for each paused trial, and these are not removed,
    # because in principle, every paused trial may be resumed in the future.
    # If checkpoints are large, this may fill up your disk.
    # Here, we use speculative checkpoint removal to keep the number of checkpoints
    # to at most ``max_num_checkpoints``. To this end, paused trials are ranked by
    # expected cost of removing their checkpoint.
    scheduler = MOBSTER(
        benchmark.config_space,
        type="promotion",
        max_resource_attr=benchmark.max_resource_attr,
        resource_attr=benchmark.resource_attr,
        mode=benchmark.mode,
        metric=benchmark.metric,
        random_seed=random_seed,
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=max_num_checkpoints,
        ),
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    # The tuner activates (early) checkpoint iff ``trial_backend.delete_checkpoints``.
    # In this case, it requests details from the scheduler (which is
    # ``early_checkpoint_removal_kwargs`` in our case).
    # Early checkpoint removal is done by appending a callback to those normally used
    # with the tuner.
    if monitor_cp_removal_in_results:
        # We can monitor how well checkpoint removal is working by storing extra results
        # (this is optional):
        extra_results_composer = CPRemovalExtraResults()
        callbacks = [
            StoreResultsCallback(extra_results_composer=extra_results_composer)
        ]
    else:
        extra_results_composer = None
        callbacks = None
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        callbacks=callbacks,
    )
    tuner.run()

    if monitor_cp_removal_in_results:
        # We have monitored how checkpoint removal has been doing over time. Here,
        # we just look at the information at the end of the experiment
        results_df = load_experiment(tuner.name).results
        final_pos = results_df.loc[:, ST_TUNER_TIME].argmax()
        final_row = dict(results_df.loc[final_pos])
        extra_results_at_end = {
            name: final_row[name] for name in extra_results_composer.keys()
        }
        logging.info(f"Extra results at end of experiment:\n{extra_results_at_end}")

    # We can obtain additional details from the callback, which is the last one
    # in ``tuner``
    callback = tuner.callbacks[-1]
    assert isinstance(callback, HyperbandRemoveCheckpointsCommon), (
        "The final callback in tuner.callbacks should be "
        f"HyperbandRemoveCheckpointsCommon, but is {type(callback)}"
    )
    trials_resumed = callback.trials_resumed_without_checkpoint()
    if trials_resumed:
        logging.info(
            f"The following {len(trials_resumed)} trials were resumed without a checkpoint:\n{trials_resumed}"
        )
    else:
        logging.info("No trials were resumed without a checkpoint")
