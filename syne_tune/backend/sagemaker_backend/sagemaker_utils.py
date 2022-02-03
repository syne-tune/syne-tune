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
import functools
import logging
import os
import re
import subprocess
import tarfile
from ast import literal_eval
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import boto3
from botocore.config import Config
from sagemaker.estimator import Framework

import syne_tune
from syne_tune.backend.trial_status import TrialResult
from syne_tune.report import retrieve
from syne_tune.util import experiment_path, random_string, \
    s3_experiment_path

logger = logging.getLogger(__name__)


def default_config() -> Config:
    # a default config that avoid throttling
    return Config(
        retries={
            'max_attempts': 10,
            'mode': 'standard'
        }
    )


def get_log(jobname: str, log_client=None) -> List[str]:
    """
    :param jobname: name of a sagemaker training job
    :param log_client: a log client, for instance `boto3.client('logs')` if None, the client is instantiated with the
    default AWS configuration
    :return: lines appearing in the log of the Sagemaker training job
    """
    if log_client is None:
        log_client = boto3.client('logs', config=default_config())
    streams = log_client.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=jobname)
    res = []

    for stream in streams['logStreams']:
        get_response = functools.partial(
            log_client.get_log_events,
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamName=stream['logStreamName'],
            startFromHead=True
        )
        response = get_response()
        for event in response['events']:
            res.append(event['message'])
        next_token = None
        while 'nextForwardToken' in response and next_token != response['nextForwardToken']:
            next_token = response['nextForwardToken']
            response = get_response(nextToken=next_token)
            for event in response['events']:
                res.append(event['message'])
    return res


def decode_sagemaker_hyperparameter(hp: str):
    # Sagemaker encodes hyperparameters as literals which are compatible with Python, except for true and false
    # that are respectively encoded as 'true' and 'false'.
    if hp == 'true':
        return True
    elif hp == 'false':
        return False
    return literal_eval(hp)


def sagemaker_search(
        trial_ids_and_names: List[Tuple[int, str]],
        sm_client=None,
) -> List[TrialResult]:
    """
    :param trial_ids_and_names: Trial ids and sagemaker jobnames to retrieve information from
    :param sm_client:
    :return: list of dictionary containing job information (status, creation-time, metrics, hyperparameters etc).
    In term of speed around 100 jobs can be retrieved per second.
    """
    if sm_client is None:
        sm_client = boto3.client(service_name='sagemaker', config=default_config())

    if len(trial_ids_and_names) == 0:
        return []

    trial_dict = {}

    # Sagemaker Search has a maximum length for filters of 20, hence we call search with 20 jobs at once
    bucket_limit = 20

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # the results of Sagemaker search are sorted by last modified time,
    # we use this dictionary to return results sorted by trial-id
    name_to_trialid_dict = {name: trialid for trialid, name in trial_ids_and_names}

    for job_chunk in chunks(trial_ids_and_names, bucket_limit):
        search_params = {
            "MaxResults": bucket_limit,
            "Resource": "TrainingJob",
            "SearchExpression": {
                "Filters": [
                    {
                        "Name": "TrainingJobName",
                        "Operator": "Equals",
                        "Value": name
                    } for (trial_id, name) in job_chunk
                ],
                "Operator": "Or"},
        }
        search_results = sm_client.search(**search_params)['Results']

        for results in search_results:
            job_info = results['TrainingJob']
            name = job_info['TrainingJobName']

            # remove sagemaker specific stuff such as container_log_level from hyperparameters
            hps = {
                k: v
                for k, v in job_info['HyperParameters'].items()
                if not k.startswith("sagemaker_")
            }

            # Sagemaker encodes hyperparameters as literals, we evaluate them to retrieve the original type
            hps = {
                k: decode_sagemaker_hyperparameter(v) for k, v in hps.items()
            }

            metrics = retrieve(log_lines=get_log(name))

            trial_id = name_to_trialid_dict[name]

            trial_dict[trial_id] = TrialResult(
                trial_id=trial_id,
                config=hps,
                metrics=metrics,
                status=job_info['TrainingJobStatus'],
                creation_time=job_info['CreationTime'],
                training_end_time=job_info.get('TrainingEndTime', None),
            )

    # Sagemaker Search returns results sorted by last modified time, we reorder the results so that they are returned
    # with the same order as the trial-ids passed
    sorted_res = [
        trial_dict[trial_id]
        for trial_id, _ in trial_ids_and_names
        if trial_id in trial_dict
    ]
    return sorted_res


def metric_definitions_from_names(metrics_names):
    """
    :param metrics_names: names of the metrics present in the log.
    Metrics must be written in the log as [metric-name]: value, for instance [accuracy]: 0.23
    :return: a list of metric dictionaries that can be passed to sagemaker so that metrics are parsed from logs, the
    list can be passed to `metric_definitions` in sagemaker.
    """
    def metric_dict(metric_name):
        """
        :param metric_name:
        :return: a sagemaker metric definition to enable Sagemaker to interpret metrics from logs
        """
        regex = rf".*[tune-metric].*\"{re.escape(metric_name)}\": ([-+]?\d\.?\d*)"
        return {'Name': metric_name, 'Regex': regex}

    return [metric_dict(m) for m in metrics_names]


def add_syne_tune_dependency(sm_estimator):
    # adds code of syne tune to the estimator to be sent with the estimator dependencies so that report.py or
    # other functions of syne tune can be found
    sm_estimator.dependencies = sm_estimator.dependencies + [str(Path(syne_tune.__path__[0]))]


def sagemaker_fit(
        sm_estimator: Framework,
        hyperparameters: Dict[str, object],
        checkpoint_s3_uri: Optional[str] = None,
        wait: bool = False,
        job_name: Optional[str] = None,
        *sagemaker_fit_args,
        **sagemaker_fit_kwargs
):
    """
    :param sm_estimator: sagemaker estimator to be fitted
    :param hyperparameters: dictionary of hyperparameters that are passed to `entry_point_script`
    :param checkpoint_s3_uri: checkpoint_s3_uri of Sagemaker Estimator
    :param wait: whether to wait for job completion
    :param metrics_names: names of metrics to track reported with `report.py`. In case those metrics are passed, their
    learning curves will be shown in Sagemaker console.
    :return: name of sagemaker job
    """
    experiment = sm_estimator
    experiment._hyperparameters = hyperparameters
    experiment.checkpoint_s3_uri = checkpoint_s3_uri

    experiment.fit(wait=wait, job_name=job_name, *sagemaker_fit_args, **sagemaker_fit_kwargs)

    return experiment.latest_training_job.job_name


def get_execution_role():
    """
    :return: sagemaker execution role that is specified with the environment variable `AWS_ROLE`, if not specified then
    we infer it by searching for the role associated to Sagemaker. Note that
    `import sagemaker; sagemaker.get_execution_role()`
    does not return the right role outside of a Sagemaker notebook.
    """
    if "AWS_ROLE" in os.environ:
        aws_role = os.environ["AWS_ROLE"]
        logger.info(f"Using Sagemaker role {aws_role} passed set as environment variable $AWS_ROLE")
        return aws_role
    else:
        logger.info(f"No Sagemaker role passed as environment variable $AWS_ROLE, inferring it.")
        client = boto3.client("iam", config=default_config())
        sm_roles = client.list_roles(PathPrefix="/service-role/")['Roles']
        for role in sm_roles:
            if 'AmazonSageMaker-ExecutionRole' in role['RoleName']:
                return role['Arn']
        raise Exception(
            "Could not infer Sagemaker role, specify it by specifying `AWS_ROLE` environement variable " \
            "or refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html to create a new one"
        )


def untar(filename: Path):
    if str(filename).endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=filename.parent)
        tar.close()


def download_sagemaker_results(s3_path: Optional[str] = None):
    """
    Download results obtained after running tuning remotely on Sagemaker,
    e.g. when using `RemoteLauncher`.
    """
    if s3_path is None:
        s3_path = s3_experiment_path()
    tgt_dir = str(experiment_path())
    cmd = f"aws s3 sync {s3_path} {tgt_dir}"
    logger.info(f"downloading sagemaker results to {tgt_dir} with command {cmd}")
    subprocess.run(cmd.split(" "))


def map_identifier_limited_length(
        name: str, max_length: int = 63, rnd_digits: int = 4) -> str:
    """
    If `name` is longer than 'max_length` characters, it is mapped to a new
    identifier of length `max_length`, being the concatenation of the first
    `max_length - rnd_digits` characters of `name`, followed by a random
    string of length `hash_digits`.

    :param name: Identifier to be limited in length
    :param max_length: Maximum length for output
    :param rnd_digits: See above
    :return: See above

    """
    orig_length = len(name)
    if orig_length <= max_length:
        return name
    else:
        assert 1 < rnd_digits < max_length
        postfix = random_string(rnd_digits)
        return name[:(max_length - rnd_digits)] + postfix
