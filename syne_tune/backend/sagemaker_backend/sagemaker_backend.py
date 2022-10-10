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
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError
import numpy as np

from sagemaker import LocalSession
from sagemaker.estimator import Framework

from syne_tune.backend.trial_backend import TrialBackend
from syne_tune.constants import ST_INSTANCE_TYPE, ST_INSTANCE_COUNT, ST_CHECKPOINT_DIR
from syne_tune.util import s3_experiment_path
from syne_tune.backend.trial_status import TrialResult, Status
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    sagemaker_search,
    get_log,
    sagemaker_fit,
    metric_definitions_from_names,
    add_syne_tune_dependency,
    map_identifier_limited_length,
    s3_copy_files_recursively,
    s3_delete_files_recursively,
    default_config,
    default_sagemaker_session,
)


logger = logging.getLogger(__name__)


class SageMakerBackend(TrialBackend):
    def __init__(
        self,
        sm_estimator: Framework,
        metrics_names: Optional[List[str]] = None,
        s3_path: Optional[str] = None,
        delete_checkpoints: bool = False,
        *args,
        **sagemaker_fit_kwargs,
    ):
        """
        :param sm_estimator: sagemaker estimator to be fitted
        :param metrics_names: name of metrics passed to `report`, used to plot live curve in sagemaker (optional, only
        used for visualization purpose)
        :param s3_path: S3 base path used for checkpointing. The full path
            also involves the tuner name and the trial_id
        :param sagemaker_fit_kwargs: extra arguments that are passed to sagemaker.estimator.Framework when fitting the
        job, for instance `{'train': 's3://my-data-bucket/path/to/my/training/data'}`
        """
        assert (
            not delete_checkpoints
        ), "delete_checkpoints=True not yet supported for SageMaker backend"
        super(SageMakerBackend, self).__init__()
        self.sm_estimator = sm_estimator

        # edit the sagemaker estimator so that metrics of the user can be plotted over time by sagemaker and so that
        # the report.py code is available
        if metrics_names is None:
            metrics_names = []
        self.metrics_names = metrics_names
        self.add_metric_definitions_to_sagemaker_estimator(metrics_names)

        st_prefix = "st-"
        if self.sm_estimator.base_job_name is None:
            base_job_name = st_prefix
        else:
            base_job_name = st_prefix + self.sm_estimator.base_job_name
        # Make sure len(base_job_name) <= 63
        self.sm_estimator.base_job_name = map_identifier_limited_length(base_job_name)

        add_syne_tune_dependency(self.sm_estimator)

        self.job_id_mapping = {}
        self.sagemaker_fit_kwargs = sagemaker_fit_kwargs

        # we keep the list of jobs that were paused/stopped as Sagemaker training job status is not immediately changed
        # after stopping a job.
        self.paused_jobs = set()
        self.stopped_jobs = set()
        # Counts how often a trial has been resumed
        self.resumed_counter = dict()
        if s3_path is None:
            s3_path = s3_experiment_path()
        self.s3_path = s3_path.rstrip("/")
        self.tuner_name = None

    @property
    def sm_client(self):
        return boto3.client(service_name="sagemaker", config=default_config())

    def add_metric_definitions_to_sagemaker_estimator(self, metrics_names: List[str]):
        # We add metric definitions corresponding to the metrics passed by `report` that the user wants to track
        # this allows to plot live learning curves of metrics in Sagemaker console.
        # The reason why we ask to the user metric names is that they are required to be known before hand so that live
        # plotting works.
        if self.sm_estimator.metric_definitions is None:
            self.sm_estimator.metric_definitions = metric_definitions_from_names(
                metrics_names
            )
        else:
            self.sm_estimator.metric_definitions = (
                self.sm_estimator.metric_definitions
                + metric_definitions_from_names(self.metrics_names)
            )
        if len(self.sm_estimator.metric_definitions) > 40:
            logger.warning(
                "Sagemaker only supports 40 metrics for learning curve visualization, keeping only the first 40"
            )
            self.sm_estimator.metric_definitions = self.sm_estimator.metric_definitions[
                :40
            ]

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        res = sagemaker_search(
            trial_ids_and_names=[
                (jobid, self.job_id_mapping[jobid]) for jobid in trial_ids
            ],
            sm_client=self.sm_client,
        )

        # overrides the status return by Sagemaker as the stopping decision may not have been propagated yet.
        for trial_res in res:
            if trial_res.trial_id in self.paused_jobs:
                trial_res.status = Status.paused
            if trial_res.trial_id in self.stopped_jobs:
                trial_res.status = Status.stopped
        return res

    @staticmethod
    def _numpy_serialize(dict):
        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        return json.loads(json.dumps(dict, default=np_encoder))

    def _checkpoint_s3_uri_for_trial(self, trial_id: int) -> str:
        res_path = self.s3_path
        if self.tuner_name is not None:
            res_path = f"{res_path}/{self.tuner_name}"
        return f"{res_path}/{str(trial_id)}/checkpoints/"

    def _schedule(self, trial_id: int, config: Dict):
        config[ST_CHECKPOINT_DIR] = "/opt/ml/checkpoints"
        hyperparameters = config.copy()

        # This passes the instance type and instance count to the training function in Sagemaker as hyperparameters
        # with reserved names `st_instance_type` and `st_instance_count`.
        # We pass them as hyperparameters as it is not easy to get efficiently from inside Sagemaker training script
        # (this information is not given for instance as Sagemaker environment variables).
        # This allows to: 1) measure cost in the worker 2) tune instance_type and instance_count by having
        # `st_instance_type` or `st_instance_count` in the config space.
        # TODO once we have a multiobjective scheduler, we should add an example on how to tune instance-type/count.
        if ST_INSTANCE_TYPE not in hyperparameters:
            hyperparameters[ST_INSTANCE_TYPE] = self.sm_estimator.instance_type
        else:
            self.sm_estimator.instance_type = hyperparameters[ST_INSTANCE_TYPE]
        if ST_INSTANCE_COUNT not in hyperparameters:
            hyperparameters[ST_INSTANCE_COUNT] = self.sm_estimator.instance_count
        else:
            self.sm_estimator.instance_count = hyperparameters[ST_INSTANCE_COUNT]

        if self.sm_estimator.instance_type != "local":
            checkpoint_s3_uri = self._checkpoint_s3_uri_for_trial(trial_id)
            logging.info(
                f"Trial {trial_id} will checkpoint results to {checkpoint_s3_uri}."
            )
        else:
            # checkpointing is not supported in local mode. When using local mode with remote tuner (for instance for
            # debugging), results are not stored.
            checkpoint_s3_uri = None

        # Once a trial gets resumed, the running job number has to feature in
        # the SM job_name
        jobname = sagemaker_fit(
            sm_estimator=self.sm_estimator,
            # the encoder fixes json error "TypeError: Object of type 'int64' is not JSON serializable"
            hyperparameters=self._numpy_serialize(hyperparameters),
            checkpoint_s3_uri=checkpoint_s3_uri,
            job_name=self._make_sagemaker_jobname(
                trial_id=trial_id,
                job_running_number=self.resumed_counter.get(trial_id, 0),
            ),
            **self.sagemaker_fit_kwargs,
        )
        logger.info(f"scheduled {jobname} for trial-id {trial_id}")
        self.job_id_mapping[trial_id] = jobname

    def _make_sagemaker_jobname(self, trial_id: int, job_running_number: int) -> str:
        f"""
        :param trial_id:
        :param job_running_number: the number of times the trial was resumed
        :return: sagemaker job name with the form [trial_id]-[job_running_number]-[tuner_name]
        trial_id is put first to avoid mismatch when searching for job information in SageMaker from prefix.
        """
        job_name = f"{trial_id}"
        if job_running_number > 0:
            job_name += f"-{job_running_number}"
        job_name += f"-{self.tuner_name}"
        return job_name

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        self._stop_trial_job(trial_id)
        self.paused_jobs.add(trial_id)

    def _stop_trial(self, trial_id: int, result: Optional[dict]):
        training_job_name = self.job_id_mapping[trial_id]
        logger.info(f"stopping {trial_id} ({training_job_name})")
        self._stop_trial_job(trial_id)
        self.stopped_jobs.add(trial_id)

    def _stop_trial_job(self, trial_id: int):
        training_job_name = self.job_id_mapping[trial_id]
        try:
            self.sm_client.stop_training_job(TrainingJobName=training_job_name)
        except ClientError:
            # the scheduler may have decided to stop a job that finished already
            pass

    def _resume_trial(self, trial_id: int):
        assert (
            trial_id in self.paused_jobs
        ), f"Try to resume trial {trial_id} that was not paused before."
        self.paused_jobs.remove(trial_id)
        if trial_id in self.resumed_counter:
            self.resumed_counter[trial_id] += 1
        else:
            self.resumed_counter[trial_id] = 1

    def stdout(self, trial_id: int) -> List[str]:
        return get_log(self.job_id_mapping[trial_id])

    def stderr(self, trial_id: int) -> List[str]:
        return get_log(self.job_id_mapping[trial_id])

    @property
    def source_dir(self) -> Optional[str]:
        return self.sm_estimator.source_dir

    def set_entrypoint(self, entry_point: str):
        self.sm_estimator.entry_point = entry_point

    def entrypoint_path(self) -> Path:
        return Path(self.sm_estimator.entry_point)

    def __getstate__(self):
        # dont store sagemaker client that cannot be serialized, we could remove it by changing our interface
        # and having kwargs/args of SagemakerFramework in the constructor of this class (that would be serializable)
        # plus the class (for instance PyTorch)
        self.sm_estimator.sagemaker_session = None
        self.sm_estimator.latest_training_job = None
        self.sm_estimator.jobs = []
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        self.initialize_sagemaker_session()

        # adjust the dependencies when running Sagemaker backend on sagemaker with remote launcher
        # since they are in a different path
        is_running_on_sagemaker = "SM_OUTPUT_DIR" in os.environ
        if is_running_on_sagemaker:
            # todo support dependencies on Sagemaker estimator, one way would be to ship them with the remote
            #  dependencies
            self.sm_estimator.dependencies = [
                Path(dep).name for dep in self.sm_estimator.dependencies
            ]

    def initialize_sagemaker_session(self):
        if boto3.Session().region_name is None:
            # avoids error "Must setup local AWS configuration with a region supported by SageMaker."
            # in case no region is explicitely configured
            os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

        if self.sm_estimator.instance_type in ("local", "local_gpu"):
            if (
                self.sm_estimator.instance_type == "local_gpu"
                and self.sm_estimator.instance_count > 1
            ):
                raise RuntimeError("Distributed Training in Local GPU is not supported")
            self.sm_estimator.sagemaker_session = LocalSession()
        else:
            # Use SageMaker boto3 client with default config. This is important
            # to configure automatic retry options properly
            self.sm_estimator.sagemaker_session = default_sagemaker_session()

    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        s3_source_path = self._checkpoint_s3_uri_for_trial(src_trial_id)
        s3_target_path = self._checkpoint_s3_uri_for_trial(tgt_trial_id)
        logger.info(
            f"Copying checkpoint files from {s3_source_path} to " + s3_target_path
        )
        result = s3_copy_files_recursively(s3_source_path, s3_target_path)
        num_action_calls = result["num_action_calls"]
        if num_action_calls == 0:
            logger.info(f"No checkpoint files found at {s3_source_path}")
        else:
            num_successful_action_calls = result["num_successful_action_calls"]
            assert num_successful_action_calls == num_action_calls, (
                f"{num_successful_action_calls} files copied successfully, "
                + f"{num_action_calls - num_successful_action_calls} failures. "
                + "Error:\n"
                + result["first_error_message"]
            )

    def delete_checkpoint(self, trial_id: int):
        s3_path = self._checkpoint_s3_uri_for_trial(trial_id)
        result = s3_delete_files_recursively(s3_path)
        num_action_calls = result["num_action_calls"]
        if num_action_calls > 0:
            num_successful_action_calls = result["num_successful_action_calls"]
            if num_successful_action_calls == num_action_calls:
                logger.info(
                    f"Deleted {num_action_calls} checkpoint files from {s3_path}"
                )
            else:
                logger.warning(
                    f"Successfully deleted {num_successful_action_calls} "
                    f"checkpoint files from {s3_path}, but failed to delete "
                    f"{num_action_calls - num_successful_action_calls} files. "
                    "Error:\n" + result["first_error_message"]
                )

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        # we use the tuner-name to set the checkpoint directory
        self.tuner_name = tuner_name

    def on_tuner_save(self):
        # Re-initialize the session after `Tuner` is serialized
        self.initialize_sagemaker_session()
