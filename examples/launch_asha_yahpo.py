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
Example for running ASHA with 4 workers with the simulator backend based on three Yahpo surrogate benchmarks.
"""
import logging
from dataclasses import dataclass

from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import Domain
from syne_tune.try_import import try_import_visual_message

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(try_import_visual_message())


def plot_yahpo_learning_curves(
    trial_backend, benchmark: str, time_col: str, metric_col: str
):
    bb = trial_backend.blackbox
    plt.figure()
    plt.title(
        f"Learning curves from Yahpo {benchmark} for 10 different hyperparameters."
    )
    for i in range(10):
        config = {
            k: v.sample() if isinstance(v, Domain) else v
            for k, v in bb.configuration_space.items()
        }
        evals = bb(config)
        time_index = next(
            i for i, name in enumerate(bb.objectives_names) if name == time_col
        )
        accuracy_index = next(
            i for i, name in enumerate(bb.objectives_names) if name == metric_col
        )
        import numpy as np

        if np.diff(evals[:, time_index]).min() < 0:
            print("negative time between two different steps...")
        plt.plot(evals[:, time_index], evals[:, accuracy_index])
    plt.xlabel(time_col)
    plt.ylabel(metric_col)
    plt.show()


@dataclass
class BenchmarkInfo:
    blackbox_name: str
    elapsed_time_attr: str
    metric: str
    dataset: str
    mode: str
    max_t: int
    resource_attr: str


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    benchmark_infos = {
        "nb301": BenchmarkInfo(
            elapsed_time_attr="runtime",
            metric="val_accuracy",
            blackbox_name="yahpo-nb301",
            dataset="CIFAR10",
            mode="max",
            max_t=97,
            resource_attr="epoch",
        ),
        "lcbench": BenchmarkInfo(
            elapsed_time_attr="time",
            metric="val_accuracy",
            blackbox_name="yahpo-lcbench",
            dataset="3945",
            mode="max",
            max_t=51,
            resource_attr="epoch",
        ),
        "fcnet": BenchmarkInfo(
            elapsed_time_attr="runtime",
            metric="valid_mse",
            blackbox_name="yahpo-fcnet",
            dataset="fcnet_naval_propulsion",
            mode="min",
            max_t=99,
            resource_attr="epoch",
        ),
    }
    for benchmark in ["nb301", "lcbench", "fcnet"]:
        benchmark_info = benchmark_infos[benchmark]

        trial_backend = BlackboxRepositoryBackend(
            blackbox_name=benchmark_info.blackbox_name,
            elapsed_time_attr=benchmark_info.elapsed_time_attr,
            dataset=benchmark_info.dataset,
        )

        plot_yahpo_learning_curves(
            trial_backend,
            benchmark=benchmark,
            time_col=benchmark_info.elapsed_time_attr,
            metric_col=benchmark_info.metric,
        )

        max_resource_attr = "epochs"
        config_space = dict(
            trial_backend.blackbox.configuration_space,
            **{max_resource_attr: benchmark_info.max_t},
        )
        scheduler = ASHA(
            config_space=config_space,
            max_resource_attr=max_resource_attr,
            resource_attr=benchmark_info.resource_attr,
            mode=benchmark_info.mode,
            metric=benchmark_info.metric,
        )

        stop_criterion = StoppingCriterion(max_num_trials_started=100)

        # It is important to set ``sleep_time`` to 0 here (mandatory for simulator
        # backend)
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=4,
            sleep_time=0,
            print_update_interval=10,
            # This callback is required in order to make things work with the
            # simulator callback. It makes sure that results are stored with
            # simulated time (rather than real time), and that the time_keeper
            # is advanced properly whenever the tuner loop sleeps
            callbacks=[SimulatorCallback()],
            tuner_name=f"ASHA-Yahpo-{benchmark}",
        )
        tuner.run()

        tuning_experiment = load_experiment(tuner.name)
        tuning_experiment.plot()
