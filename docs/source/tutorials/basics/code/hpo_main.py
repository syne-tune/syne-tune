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
import logging
from argparse import ArgumentParser
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint, uniform, loguniform


# Configuration space (or search space)
config_space = {
    "n_units_1": randint(4, 1024),
    "n_units_2": randint(4, 1024),
    "batch_size": randint(8, 128),
    "dropout_1": uniform(0, 0.99),
    "dropout_2": uniform(0, 0.99),
    "learning_rate": loguniform(1e-6, 1),
    "weight_decay": loguniform(1e-8, 1),
}


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # [1]
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "RS",
            "BO",
            "ASHA-STOP",
            "ASHA-PROM",
            "MOBSTER-STOP",
            "MOBSTER-PROM",
        ),
        default="RS",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        default=3 * 3600,
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="basic-tutorial",
    )
    args, _ = parser.parse_known_args()

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training script
    if args.method in ("RS", "BO"):
        train_file = "traincode_report_end.py"
    elif args.method.endswith("STOP"):
        train_file = "traincode_report_eachepoch.py"
    else:
        train_file = "traincode_report_withcheckpointing.py"
    entry_point = Path(__file__).parent / train_file
    max_resource_level = 81  # Maximum number of training epochs
    mode = "max"
    metric = "accuracy"
    resource_attr = "epoch"
    max_resource_attr = "epochs"

    # Additional fixed parameters  [2]
    config_space.update(
        {
            max_resource_attr: max_resource_level,
            "dataset_path": "./",
        }
    )

    # Local backend: Responsible for scheduling trials  [3]
    # The local backend runs trials as sub-processes on a single instance
    trial_backend = LocalBackend(entry_point=str(entry_point))

    # Scheduler: Depends on `args.method`  [4]
    scheduler = None
    # Common scheduler kwargs
    method_kwargs = dict(
        metric=metric,
        mode=mode,
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
        search_options={"num_init_random": args.n_workers + 2},
    )
    sch_type = "promotion" if args.method.endswith("PROM") else "stopping"
    if args.method == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs)
    elif args.method == "BO":
        scheduler = BayesianOptimization(config_space, **method_kwargs)
    else:
        # Multi-fidelity method
        method_kwargs["resource_attr"] = resource_attr
        if args.method.startswith("ASHA"):
            scheduler = ASHA(config_space, type=sch_type, **method_kwargs)
        elif args.method.startswith("MOBSTER"):
            scheduler = MOBSTER(config_space, type=sch_type, **method_kwargs)
        else:
            raise NotImplementedError(args.method)

    # Stopping criterion: We stop after `args.max_wallclock_time` seconds
    # [5]
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_wallclock_time)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        tuner_name=args.experiment_tag,
        metadata={
            "seed": args.random_seed,
            "algorithm": args.method,
            "tag": args.experiment_tag,
        },
    )

    tuner.run()
