"""
Example showing how to tune instance types and hyperparameters with a Sagemaker Framework.
"""
import logging
from pathlib import Path

from sagemaker.huggingface import HuggingFace

from sagemaker_tune.backend.sagemaker_backend.instance_info import select_instance_type
from sagemaker_tune.backend.sagemaker_backend.sagemaker_backend import SagemakerBackend
from sagemaker_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from sagemaker_tune.constants import SMT_WORKER_TIME, SMT_WORKER_COST
from sagemaker_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from sagemaker_tune.remote.remote_launcher import RemoteLauncher
from sagemaker_tune.stopping_criterion import StoppingCriterion
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.search_space import loguniform, choice

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 2
    epochs = 4

    # Select the instance types that are searched.
    # Alternatively, you can define the instance list explicitly: `instance_types = ['ml.c5.xlarge', 'ml.m5.2xlarge']`
    instance_types = select_instance_type(min_gpu=1, max_cost_per_hour=5.0)

    print(f"tuning over hyperparameters and instance types: {instance_types}")

    # define a search space that contains hyperparameters (learning-rate, weight-decay) and instance-type.
    config_space = {
        'smt_instance_type': choice(instance_types),
        'learning_rate': loguniform(1e-6, 1e-4),
        'weight_decay': loguniform(1e-5, 1e-2),
        'epochs': epochs,
        'dataset_path': './',
    }
    entry_point = Path(__file__).parent / "training_scripts" / "distilbert_on_imdb" / "distilbert_on_imdb.py"
    metric = "accuracy"

    # Define a MOASHA scheduler that searches over the config space to maximise accuracy and minimize cost and time.
    scheduler = MOASHA(
        max_t=epochs,
        time_attr="step",
        metrics=[metric, SMT_WORKER_COST, SMT_WORKER_TIME],
        mode=['max', 'min', 'min'],
        config_space=config_space,
    )

    # Define the training function to be tuned, use the Sagemaker backend to execute trials as separate training job
    # (since they are quite expensive).
    backend = SagemakerBackend(
        sm_estimator=HuggingFace(
            entry_point=str(entry_point),
            base_job_name='hpo-transformer',
            # instance-type given here are override by Sagemaker tune with values sampled from `smt_instance_type`.
            instance_type='ml.m5.large',
            instance_count=1,
            transformers_version='4.4',
            pytorch_version='1.6',
            py_version='py36',
            max_run=3600,
            role=get_execution_role(),
            dependencies=[str(Path(__file__).parent.parent / "benchmarks/")],
        ),
    )

    remote_launcher = RemoteLauncher(
        tuner=Tuner(
            backend=backend,
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(max_wallclock_time=3600, max_cost=10.0),
            n_workers=n_workers,
            sleep_time=5.0,
        ),
        dependencies=[str(Path(__file__).parent.parent / "benchmarks/")],
    )

    remote_launcher.run(wait=False)
