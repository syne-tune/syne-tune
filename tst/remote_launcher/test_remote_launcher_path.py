# # Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License").
# # You may not use this file except in compliance with the License.
# # A copy of the License is located at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # or in the "license" file accompanying this file. This file is distributed
# # on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# # express or implied. See the License for the specific language governing
# # permissions and limitations under the License.
# import logging
# from pathlib import Path
#
# import pytest
# from sagemaker.pytorch import PyTorch
#
# from syne_tune.backend import SagemakerBackend
# from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
# from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
# from syne_tune.remote.remote_launcher import RemoteLauncher
# from syne_tune import StoppingCriterion
# from syne_tune import Tuner
#
# root = Path(__file__).parent
# sm_estimator = PyTorch(
#     entry_point="folder2/main.py",
#     source_dir=str(root / "folder1"),
#     instance_type="local",
#     instance_count=1,
#     py_version="py3",
#     framework_version="1.7.1",
#     role="dummy",
# )
#
# backend = SagemakerBackend(sm_estimator=sm_estimator)
# remote_launcher = RemoteLauncher(
#     tuner=Tuner(
#         backend=backend,
#         scheduler=FIFOScheduler({}, searcher='random', metric="dummy"),
#         stop_criterion=StoppingCriterion(max_wallclock_time=600),
#         n_workers=4,
#     )
# )
# remote_launcher.prepare_upload()
#
#
# def test_check_paths():
#     # for now, we only check that sm_estimator source_dir, endpoint script is correct
#     # todo check that dependencies are correct
#     remote_sm_estimator = remote_launcher.tuner.backend.sm_estimator
#
#     assert remote_sm_estimator.source_dir == "tuner"
#     assert (remote_launcher.upload_dir() / "folder2" / "main.py").exists()
#     assert (remote_launcher.upload_dir() / "requirements.txt").exists()
#     assert (remote_launcher.upload_dir() / "tuner.dill").exists()
#
#
# @pytest.mark.skip("this test is skipped currently as it takes ~15s and requires docker installed locally.")
# def test_estimator():
#     tuner = Tuner.load(remote_launcher.upload_dir())
#     remote_sm_estimator = tuner.backend.sm_estimator
#     remote_sm_estimator.source_dir = str(remote_launcher.upload_dir())
#     remote_sm_estimator.fit()