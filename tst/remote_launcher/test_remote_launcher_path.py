import logging
from pathlib import Path

import pytest
from sagemaker.pytorch import PyTorch

from sagemaker_tune.backend.sagemaker_backend.sagemaker_backend import SagemakerBackend
from sagemaker_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from sagemaker_tune.optimizer.schedulers.fifo import FIFOScheduler
from sagemaker_tune.remote.remote_launcher import RemoteLauncher
from sagemaker_tune.stopping_criterion import StoppingCriterion
from sagemaker_tune.tuner import Tuner

root = Path(__file__).parent
sm_estimator = PyTorch(
    entry_point="folder2/main.py",
    source_dir=str(root / "folder1"),
    instance_type="local",
    instance_count=1,
    py_version="py3",
    framework_version="1.7.1",
    role=get_execution_role(),
)

backend = SagemakerBackend(sm_estimator=sm_estimator)
remote_launcher = RemoteLauncher(
    tuner=Tuner(
        backend=backend,
        scheduler=FIFOScheduler({}, searcher='random', metric="dummy"),
        stop_criterion=StoppingCriterion(max_wallclock_time=600),
        n_workers=4,
    )
)
remote_launcher.prepare_upload()


def test_check_paths():
    # for now, we only check that sm_estimator source_dir, endpoint script is correct
    # todo check that dependencies are correct
    remote_sm_estimator = remote_launcher.tuner.backend.sm_estimator

    assert remote_sm_estimator.source_dir == "tuner"
    assert (remote_launcher.upload_dir() / "folder2" / "main.py").exists()
    assert (remote_launcher.upload_dir() / "requirements.txt").exists()
    assert (remote_launcher.upload_dir() / "tuner.dill").exists()


@pytest.mark.skip("this test is skipped currently as it takes ~15s and requires docker installed locally.")
def test_estimator():
    tuner = Tuner.load(remote_launcher.upload_dir())
    remote_sm_estimator = tuner.backend.sm_estimator
    remote_sm_estimator.source_dir = str(remote_launcher.upload_dir())
    remote_sm_estimator.fit()