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
from typing import Optional
from pathlib import Path

from syne_tune.util import s3_experiment_path
from syne_tune.try_import import try_import_aws_message

try:
    from sagemaker.pytorch import PyTorch
    from sagemaker.huggingface import HuggingFace
    from sagemaker.mxnet import MXNet
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.sklearn import SKLearn
    from sagemaker.chainer import Chainer
    from sagemaker.xgboost import XGBoost
except ImportError:
    print(try_import_aws_message())


sagemaker_estimator = {
    "PyTorch": PyTorch,
    "HuggingFace": HuggingFace,
    "MXNet": MXNet,
    "TensorFlow": TensorFlow,
    "SKLearn": SKLearn,
    "Chainer": Chainer,
    "XGBoost": XGBoost,
}


# Used for simulator back-end experiments and for remote launching of
# SageMaker back-end experiments
def basic_cpu_instance_sagemaker_estimator(**kwargs):
    """
    Returns SageMaker estimator to be used for simulator back-end experiments
    and for remote launching of SageMaker back-end experiments.

    :param kwargs: Extra arguments to SageMaker estimator
    :return: SageMaker estimator
    """
    # return SKLearn(
    #     instance_type="ml.c5.4xlarge",
    #     instance_count=1,
    #     py_version="py3",
    #     framework_version="1.0-1",
    #     **kwargs,
    # )
    return PyTorch(
        instance_type="ml.c5.4xlarge",
        instance_count=1,
        py_version="py38",
        framework_version="1.12.1",
        **kwargs,
    )


def filter_none(a: dict) -> dict:
    return {k: v for k, v in a.items() if v is not None}


def message_sync_from_s3(experiment_tag: str) -> str:
    return (
        "Launched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_tag)} "
        f'~/syne-tune/{experiment_tag}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )


def combine_requirements_txt(synetune_requirements_file: Path, script: Path) -> Path:
    script_requirements_file = script.parent / "requirements.txt"
    target_requirements_file = synetune_requirements_file.parent / "requirements.txt"
    f1_exists = synetune_requirements_file.exists()
    f2_exists = script_requirements_file.exists()
    equal_fnames = synetune_requirements_file == target_requirements_file
    assert not (f1_exists and f2_exists and equal_fnames), (
        f"{synetune_requirements_file} would be overwritten by combined "
        "requirements.txt file. Please use a different filename"
    )
    if f1_exists:
        if not f2_exists and equal_fnames:
            return target_requirements_file  # Nothing to do
        target_content = synetune_requirements_file.read_text()
    else:
        target_content = ""
    if f2_exists:
        extra_content = script_requirements_file.read_text()
        target_content += "\n" + extra_content
    if target_content:
        target_requirements_file.write_text(target_content)
    return target_requirements_file


def ERR_MSG(fname: str) -> str:
    return (
        f"I'll create a safe '{fname}' for you:\n"
        "   syne-tune[extra]\n"
        "   tqdm\n"
        "You can optimize start times by using restricted dependencies, e.g.:\n"
        "   syne-tune[gpsearchers,kde]\n"
        "   tqdm\n"
        "If your launcher script requires additional dependencies, add them "
        "here as well."
    )


def find_or_create_requirements_txt(
    entry_point: Path, requirements_fname: Optional[str] = None
) -> Path:
    if requirements_fname is None:
        requirements_fname = "requirements.txt"
    files = list(entry_point.parent.rglob("requirements*.txt"))
    if len(files) == 0:
        print(
            f"Could not find {entry_point.parent}/requirements*.txt\n"
            + ERR_MSG(requirements_fname)
        )
        fname = entry_point.parent / requirements_fname
        fname.write_text("syne-tune[extra]\ntqdm\n")
    elif len(files) > 1:
        # Filter out the typical target fname
        target_fname = entry_point.parent / "requirements.txt"
        files = [fname for fname in files if fname != target_fname]
        fname = files[0]
        assert (
            len(files) == 1
        ), f"Found more than one {entry_point.parent}/requirements*.txt:\n" + str(
            [str(path) for path in files]
        )
    else:
        fname = files[0]
    return fname
