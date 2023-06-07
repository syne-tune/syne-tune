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
from typing import List, Optional, Tuple, Dict, Any
import boto3
from botocore.exceptions import ClientError
import time

from sagemaker import LocalSession
from sagemaker.estimator import Framework

from syne_tune.backend.trial_backend import TrialBackend, BUSY_STATUS
from syne_tune.constants import (
    ST_INSTANCE_TYPE,
    ST_INSTANCE_COUNT,
    ST_CHECKPOINT_DIR,
    ST_CONFIG_JSON_FNAME_ARG,
)
from syne_tune.util import s3_experiment_path, dump_json_with_numpy
from syne_tune.backend.trial_status import TrialResult, Status
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    sagemaker_search,
    get_log,
    sagemaker_fit,
    add_syne_tune_dependency,
    map_identifier_limited_length,
    s3_copy_objects_recursively,
    s3_delete_objects_recursively,
    default_config,
    default_sagemaker_session,
    add_metric_definitions_to_sagemaker_estimator,
)


logger = logging.getLogger(__name__)


CONFIG_JSON_FILENAME = "syne_tune_sm_backend_config_31415927"


class SageMakerBackend(TrialBackend):
    """
    This backend executes each trial evaluation as a separate SageMaker
    training job, using ``sm_estimator`` as estimator.

    Checkpoints are written to and loaded from S3, using ``checkpoint_s3_uri``
    of the estimator.

    Compared to :class:`LocalBackend`, this backend can run any number of
    jobs in parallel (given sufficient resources), and any instance type can
    be used.

    This backend allows to select the instance type and count for a trial
    evaluation, by passing values in the configuration, using names
    :const:`~syne_tune.constants.ST_INSTANCE_TYPE` and
    :const:`~syne_tune.constants.ST_INSTANCE_COUNT`. If these are given in the
    configuration, they overwrite the default in ``sm_estimator``. This allows
    for tuning instance type and count along with the hyperparameter
    configuration.

    Additional arguments on top of parent class
    :class:`~syne_tune.backend.trial_backend.TrialBackend`:

    :param sm_estimator: SageMaker estimator for trial evaluations.
    :param metrics_names: Names of metrics passed to ``report``, used to plot
        live curve in SageMaker (optional, only used for visualization)
    :param s3_path: S3 base path used for checkpointing. The full path
        also involves the tuner name and the ``trial_id``. The default base
        path is the S3 bucket associated with the SageMaker account
    :param sagemaker_fit_kwargs: Extra arguments that passed to
        :class:`sagemaker.estimator.Framework` when fitting the job, for instance
        :code:`{'train': 's3://my-data-bucket/path/to/my/training/data'}`
    """

    def __init__(
        self,
        sm_estimator: Framework,
        metrics_names: Optional[List[str]] = None,
        s3_path: Optional[str] = None,
        delete_checkpoints: bool = False,
        pass_args_as_json: bool = False,
        **sagemaker_fit_kwargs,
    ):
        super(SageMakerBackend, self).__init__(
            delete_checkpoints=delete_checkpoints, pass_args_as_json=pass_args_as_json
        )
        self.sm_estimator = sm_estimator

        # edit the sagemaker estimator so that metrics of the user can be plotted over time by sagemaker and so that
        # the report.py code is available
        if metrics_names is None:
            metrics_names = []
        self.add_metric_definitions_to_sagemaker_estimator(metrics_names)

        st_prefix = "st-"
        if self.sm_estimator.base_job_name is None:
            base_job_name = st_prefix
        else:
            base_job_name = st_prefix + self.sm_estimator.base_job_name
        # Make sure len(base_job_name) <= 63
        self.sm_estimator.base_job_name = map_identifier_limited_length(base_job_name)

        add_syne_tune_dependency(self.sm_estimator)

        self.job_id_mapping = dict()
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
        # Trials which may currently be busy (status in ``BUSY_STATUS``). The
        # corresponding jobs are polled for status in ``busy_trial_ids``, and
        # new trials are addd in :meth:`_schedule`.
        # Note: A trial can be in ``paused_jobs`` or ``stopped_jobs`` and still
        # be busy, because the underlying SM training job is still not completed
        self._busy_trial_id_candidates = set()
        # This is to estimate the stopping time for a trial (useful information
        # for now, can be removed once stop delays are reduced).
        # Note: Trials with a very short stop delay may be missed. This is fine,
        # because we mainly want to highlight long stop delays.
        self._stopping_time = dict()
        # Collects trial IDs for which checkpoints have been deleted (see
        # :meth:`delete_checkpoint`)
        self._trial_ids_deleted_checkpoints = set()

    @property
    def sm_client(self):
        return boto3.client(service_name="sagemaker", config=default_config())

    def add_metric_definitions_to_sagemaker_estimator(self, metrics_names: List[str]):
        # We add metric definitions corresponding to the metrics passed by ``report`` that the user wants to track
        # this allows to plot live learning curves of metrics in Sagemaker console.
        # The reason why we ask to the user metric names is that they are required to be known before hand so that live
        # plotting works.
        add_metric_definitions_to_sagemaker_estimator(self.sm_estimator, metrics_names)

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        trial_ids_and_names = []
        for jobid in trial_ids:
            name = self.job_id_mapping.get(jobid)
            if name is not None:
                trial_ids_and_names.append((jobid, name))
        if trial_ids_and_names:
            res = sagemaker_search(
                trial_ids_and_names=trial_ids_and_names,
                sm_client=self.sm_client,
            )
        else:
            res = []

        # overrides the status return by Sagemaker as the stopping decision may not have been propagated yet.
        for trial_res in res:
            trial_id = trial_res.trial_id
            if trial_id in self.paused_jobs:
                trial_res.status = Status.paused
            if trial_id in self.stopped_jobs:
                trial_res.status = Status.stopped
        return res

    @staticmethod
    def _numpy_serialize(mydict):
        return json.loads(dump_json_with_numpy(mydict))

    def _checkpoint_s3_uri_for_trial(self, trial_id: int) -> str:
        res_path = self.s3_path
        if self.tuner_name is not None:
            res_path = f"{res_path}/{self.tuner_name}"
        return f"{res_path}/{str(trial_id)}/checkpoints/"

    def _config_json_filename(self, trial_id: int, with_path: bool) -> str:
        fname = CONFIG_JSON_FILENAME + f"_{trial_id}.json"
        if with_path and self.source_dir is not None:
            return str(Path(self.source_dir) / fname)
        else:
            return fname

    def _hyperparameters_from_config(
        self, trial_id: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares hyperparameters, to be sent to the entry point script as
        command line arguments, given the configuration ``config``. If
        ``pass_args_as_json == False``, this is just a copy of ``config``.

        But otherwise, the configuration is written to a JSON file, whose
        name becomes a hyperparameter, but entries of the config are not
        hyperparameters. Note that some default entries attached to the
        config by Syne Tune are always passed as command line arguments, so if
        ``pass_args_as_json == True``, they are removed from the config before
        this is written as JSON file.

        :param trial_id: ID of trial
        :param config: Configuration
        :return: Hyperparameters to be passed to estimator entry point
        """
        config_copy = config.copy()
        if not self.pass_args_as_json:
            return config_copy
        else:
            self._set_source_dir()  # Make sure that ``source_dir`` attribute is set
            result = self._prepare_hyperparameters_if_args_as_json(
                trial_id, config_copy
            )
            dump_json_with_numpy(
                config_copy, self._config_json_filename(trial_id, with_path=True)
            )
            return result

    def _set_source_dir(self):
        if self.source_dir is None:
            entrypoint_path = self.entrypoint_path()
            source_dir = str(entrypoint_path.parent)
            entrypoint_name = entrypoint_path.name
            logger.warning(
                "sm_estimator.source_dir is not set, but is needed for "
                "pass_args_as_json == True. Setting them to:\n"
                f"source_dir = {source_dir}, entry_point = {entrypoint_name}"
            )
            self.sm_estimator.source_dir = source_dir
            self.sm_estimator.entry_point = entrypoint_name

    def _prepare_hyperparameters_if_args_as_json(
        self, trial_id: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # The filename depends on the trial ID. Otherwise, there would be
        # clashes between trials which run at overlapping times
        result = {
            ST_CONFIG_JSON_FNAME_ARG: "./"
            + self._config_json_filename(trial_id, with_path=False)
        }
        # These arguments remain command line parameters
        if ST_INSTANCE_TYPE in config:
            result[ST_INSTANCE_TYPE] = config.pop(ST_INSTANCE_TYPE)
        if ST_INSTANCE_COUNT in config:
            result[ST_INSTANCE_COUNT] = config.pop(ST_INSTANCE_COUNT)
        return result

    def _schedule(self, trial_id: int, config: Dict[str, Any]):
        hyperparameters = self._hyperparameters_from_config(trial_id, config)
        hyperparameters[ST_CHECKPOINT_DIR] = "/opt/ml/checkpoints"

        # This passes the instance type and instance count to the training function in Sagemaker as hyperparameters
        # with reserved names ``st_instance_type`` and ``st_instance_count``.
        # We pass them as hyperparameters as it is not easy to get efficiently from inside Sagemaker training script
        # (this information is not given for instance as Sagemaker environment variables).
        # This allows to: 1) measure cost in the worker 2) tune instance_type and instance_count by having
        # ``st_instance_type`` or ``st_instance_count`` in the config space.
        # TODO once we have a multiobjective scheduler, we should add an example on how to tune instance-type/count.
        if ST_INSTANCE_TYPE not in config:
            hyperparameters[ST_INSTANCE_TYPE] = self.sm_estimator.instance_type
        else:
            self.sm_estimator.instance_type = config[ST_INSTANCE_TYPE]
        if ST_INSTANCE_COUNT not in config:
            hyperparameters[ST_INSTANCE_COUNT] = self.sm_estimator.instance_count
        else:
            self.sm_estimator.instance_count = config[ST_INSTANCE_COUNT]

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
        try:
            jobname = sagemaker_fit(
                sm_estimator=self.sm_estimator,
                hyperparameters=self._numpy_serialize(hyperparameters),
                checkpoint_s3_uri=checkpoint_s3_uri,
                job_name=self._make_sagemaker_jobname(
                    trial_id=trial_id,
                    job_running_number=self.resumed_counter.get(trial_id, 0),
                ),
                **self.sagemaker_fit_kwargs,
            )
        except ClientError as ex:
            if "ResourceLimitExceeded" in str(ex):
                logger.warning(
                    "Your resource limit has been exceeded. Here are some hints:\n"
                    "- Choose Tuner.n_workers <= your limit for the instance type\n"
                    "- Use Tuner.start_jobs_without_delay = False. Setting this to "
                    "True (default) means that more than Tuner.n_workers jobs "
                    "will run at certain times"
                )
            raise
        logger.info(f"scheduled {jobname} for trial-id {trial_id}")
        self.job_id_mapping[trial_id] = jobname
        self._busy_trial_id_candidates.add(trial_id)  # Mark trial as busy

    def _make_sagemaker_jobname(self, trial_id: int, job_running_number: int) -> str:
        """
        :param trial_id: ID of trial
        :param job_running_number: Number of times the trial was resumed
        :return: sagemaker job name with the form
            ``[trial_id]-[job_running_number]-[tuner_name]``. ``trial_id`` is put
             first to avoid mismatch when searching for job information in
            SageMaker from prefix.
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

    def _get_busy_trial_ids(
        self, trial_results: List[TrialResult]
    ) -> List[Tuple[int, str]]:
        busy_list = []
        reported_trial_ids = set()
        for result in trial_results:
            trial_id, status = result.trial_id, result.status
            reported_trial_ids.add(trial_id)
            if status in BUSY_STATUS:
                busy_list.append((trial_id, result.status))
                if status == Status.stopping and trial_id not in self._stopping_time:
                    # First time we see ``Status.stopping`` for this ``trial_id``
                    self._stopping_time[trial_id] = time.time()
            elif trial_id in self._stopping_time:
                # Trial just stopped being busy
                stop_time = time.time() - self._stopping_time[trial_id]
                logger.info(
                    f"Estimated stopping delay for trial_id {trial_id}: {stop_time:.2f} secs"
                )
                del self._stopping_time[trial_id]
        # Note: It can happen that the result of ``sagemaker_search`` does
        # not contain all trial_id's requested. We keep such trial_id's in
        # the busy list
        extra_trial_ids = []
        for trial_id in self._busy_trial_id_candidates.difference(reported_trial_ids):
            # Assume that status is "in_progress": If ``sagemaker_search``
            # drops jobs, they are the ones that have just been started
            busy_list.append((trial_id, Status.in_progress))
            extra_trial_ids.append(trial_id)
        if extra_trial_ids:
            logger.info(
                f"Did not obtain status for these trial ids: [{extra_trial_ids}]. "
                f"Will count them as busy with status {Status.in_progress}"
            )
        return busy_list

    def busy_trial_ids(self) -> List[Tuple[int, str]]:
        # Note that at this point, ``self._busy_trial_id_candidates`` contains
        # trials whose jobs have been busy in the past, but they may have
        # stopped or terminated since. We query the current status for all
        # these jobs and update ``self._busy_trial_id_candidates`` accordingly.
        # It can happen that the status for such a trial is not returned (if
        # it has just been started). In this case, the trial is kept in the
        # list and treated as busy.
        if self._busy_trial_id_candidates:
            trial_ids_and_names = [
                (trial_id, self.job_id_mapping[trial_id])
                for trial_id in self._busy_trial_id_candidates
            ]
            # This is calling the SageMaker API in order to query the current
            # status for all trials in ``_busy_trial_id_candidates``
            trial_results = sagemaker_search(
                trial_ids_and_names, sm_client=self.sm_client
            )
            busy_list = self._get_busy_trial_ids(trial_results)
            # Update internal candidate list
            self._busy_trial_id_candidates = set(trial_id for trial_id, _ in busy_list)
            return busy_list
        else:
            return []

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
        if self.source_dir is None:
            return Path(self.sm_estimator.entry_point)
        else:
            return Path(self.source_dir) / self.sm_estimator.entry_point

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
        result = s3_copy_objects_recursively(s3_source_path, s3_target_path)
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
        if trial_id in self._trial_ids_deleted_checkpoints:
            return
        s3_path = self._checkpoint_s3_uri_for_trial(trial_id)
        result = s3_delete_objects_recursively(s3_path)
        self._trial_ids_deleted_checkpoints.add(trial_id)
        num_action_calls = result["num_action_calls"]
        if num_action_calls <= 0:
            return
        num_successful_action_calls = result["num_successful_action_calls"]
        if num_successful_action_calls == num_action_calls:
            logger.info(
                f"Deleted {num_action_calls} checkpoint files for "
                f"trial_id {trial_id} from {s3_path}"
            )
        else:
            logger.warning(
                f"Successfully deleted {num_successful_action_calls} "
                f"checkpoint files for trial_id {trial_id} from "
                f"{s3_path}, but failed to delete "
                f"{num_action_calls - num_successful_action_calls} "
                "files. Error:\n" + result["first_error_message"]
            )

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        # we use the tuner-name to set the checkpoint directory
        self.tuner_name = tuner_name

    def on_tuner_save(self):
        # Re-initialize the session after :class:`~syne_tune.Tuner` is serialized
        self.initialize_sagemaker_session()

    def _cleanup_after_trial(self, trial_id: int):
        if self.pass_args_as_json:
            filename = self._config_json_filename(trial_id, with_path=True)
            Path(filename).unlink(missing_ok=True)
