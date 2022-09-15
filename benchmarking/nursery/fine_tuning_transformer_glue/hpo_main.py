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
import argparse
from pathlib import Path

from syne_tune.config_space import uniform, loguniform, choice, randint
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    ASHA,
    MOBSTER,
    BayesianOptimization,
    RandomSearch,
)
from syne_tune import (
    Tuner,
    StoppingCriterion,
)


# Different GLUE tasks and their metric names
TASK2METRICSMODE = {
    "cola": {"metric": "matthews_correlation", "mode": "max"},
    "mnli": {"metric": "accuracy", "mode": "max"},
    "mrpc": {"metric": "f1", "mode": "max"},
    "qnli": {"metric": "accuracy", "mode": "max"},
    "qqp": {"metric": "f1", "mode": "max"},
    "rte": {"metric": "accuracy", "mode": "max"},
    "sst2": {"metric": "accuracy", "mode": "max"},
    "stsb": {"metric": "spearmanr", "mode": "max"},
    "wnli": {"metric": "accuracy", "mode": "max"},
}


# Pre-trained models from HuggingFace zoo considered here
PRETRAINED_MODELS = [
    "bert-base-cased",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "roberta-base",
    "albert-base-v2",
    "distilroberta-base",
    "xlnet-base-cased",
    "albert-base-v1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="rte", choices=list(TASK2METRICSMODE.keys())
    )
    parser.add_argument(
        "--model_type", type=str, default="bert-base-cased", choices=PRETRAINED_MODELS
    )
    parser.add_argument("--max_runtime", type=int, default=1800)
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="number of epochs to train the networks",
    )
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument(
        "--optimizer", type=str, default="asha", choices=("rs", "bo", "asha", "mobster")
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="experiment name",
    )
    parser.add_argument("--choose_model", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train_valid_fraction", type=float, default=0.7)
    parser.add_argument("--store_logs_checkpoints_to_s3", type=int, default=0)

    args, _ = parser.parse_known_args()
    args.choose_model = bool(args.choose_model)
    args.store_logs_checkpoints_to_s3 = bool(args.store_logs_checkpoints_to_s3)
    return args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    dataset = args.dataset
    model_type = args.model_type
    num_train_epochs = args.num_train_epochs
    seed = args.seed
    optimizer = args.optimizer

    # Path to training script. We also need to specify the names of metrics reported
    # back from this script
    entry_point = "./run_glue_modified.py"
    metric = "eval_" + TASK2METRICSMODE[dataset]["metric"]
    mode = TASK2METRICSMODE[dataset]["mode"]
    resource_attribute = "epoch"

    # The configuration space contains all hyperparameters we would like to optimize,
    # and their search ranges.
    hyperparameter_space = {
        "learning_rate": loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": randint(16, 48),
        "warmup_ratio": uniform(0, 0.5),
    }

    # Additionally, it contains fixed parameters passed to the training script.
    # We use `save_strategy="no"` in order to save disk space. This must be
    # changed to "epoch" for HPO methods which require checkpointing.
    fixed_parameters = {
        "num_train_epochs": num_train_epochs,
        "model_name_or_path": model_type,
        "task_name": dataset,
        "train_valid_fraction": args.train_valid_fraction,
        "seed": seed,
        "do_train": True,
        "max_seq_length": 128,
        "output_dir": "tmp/" + dataset,
        "evaluation_strategy": "epoch",
        "save_strategy": "no",  # change to "epoch" if checkpoints are needed!
        "save_total_limit": 1,
    }

    config_space = {**hyperparameter_space, **fixed_parameters}

    # This is the default configuration provided by Hugging Face. It will always
    # be evaluated first
    default_configuration = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "warmup_ratio": 0.0,
    }

    # Combine HPO with model selection:
    # Just another categorical hyperparameter
    if args.choose_model:
        config_space["model_name_or_path"] = choice(PRETRAINED_MODELS)
        default_configuration["model_name_or_path"] = model_type

    # The backend is responsible to start and stop training evaluations. Here, we
    # use the local backend, which runs on a single instance
    backend = LocalBackend(entry_point=entry_point)

    # HPO algorithm
    # We can choose from these optimizers:
    schedulers = {
        "rs": RandomSearch,
        "bo": BayesianOptimization,
        "asha": ASHA,
        "mobster": MOBSTER,
    }
    scheduler_kwargs = dict(
        metric=metric,
        mode=mode,
        random_seed=seed,  # same seed passed to training function
        points_to_evaluate=[default_configuration],  # evaluate this one first
    )
    if optimizer in {"asha", "mobster"}:
        # The multi-fidelity methods need extra information
        scheduler_kwargs["resource_attr"] = resource_attribute
        # Maximum resource level information in `config_space`:
        scheduler_kwargs["max_resource_attr"] = "num_train_epochs"
    scheduler = schedulers[optimizer](config_space, **scheduler_kwargs)

    # All parts come together in the tuner, which runs the experiment
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_runtime)
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        metadata=vars(args),  # metadata is stored along with results
        tuner_name=args.experiment_name,
    )

    # Set path for logs and checkpoints
    if args.store_logs_checkpoints_to_s3:
        backend.set_path(results_root=tuner.tuner_path)
    else:
        backend.set_path(
            results_root=str(Path("~/").expanduser()), tuner_name=tuner.name
        )

    tuner.run()  # off we go!
