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

DEFAULT_CPU_INSTANCE = "ml.c5.4xlarge"
DEFAULT_CPU_INSTANCE_SMALL = "ml.m5.large"
DEFAULT_GPU_INSTANCE_1GPU = "ml.g4dn.xlarge"
DEFAULT_GPU_INSTANCE_4GPU = "ml.g4dn.12xlarge"

PYTORCH_LATEST_FRAMEWORK = "1.12.1"
PYTORCH_LATEST_PY_VERSION = "py38"

HUGGINGFACE_LATEST_FRAMEWORK_VERSION = "4.4"
HUGGINGFACE_LATEST_PYTORCH_VERSION = "1.7.1"
HUGGINGFACE_LATEST_TRANSFORMERS_VERSION = "4.6.1"
HUGGINGFACE_LATEST_PY_VERSION = "py36"

MXNET_LATEST_PY_VERSION = "py38"
MXNET_LATEST_VERSION = "1.9.0"


def instance_sagemaker_estimator(**kwargs):
    """
    Returns SageMaker estimator to be used for simulator back-end experiments
    and for remote launching of SageMaker back-end experiments.

    :param kwargs: Extra arguments to SageMaker estimator
    :return: SageMaker estimator
    """
    return pytorch_estimator(
        **kwargs,
    )


def basic_cpu_instance_sagemaker_estimator(**kwargs):
    """
    Returns SageMaker estimator to be used for simulator back-end experiments
    and for remote launching of SageMaker back-end experiments.

    :param kwargs: Extra arguments to SageMaker estimator
    :return: SageMaker estimator
    """
    return pytorch_estimator(
        instance_type=DEFAULT_CPU_INSTANCE,
        instance_count=1,
        **kwargs,
    )


def pytorch_estimator(**estimator_kwargs) -> PyTorch:
    """
    Get the PyTorch sagemaker estimator with the most up-to-date framework version.
    List of available containers: https://github.com/aws/deep-learning-containers/blob/master/available_images.md

    :param estimator_kwargs: Estimator parameters as discussed in
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
    :return: PyTorch estimator
    """
    return PyTorch(
        py_version=PYTORCH_LATEST_PY_VERSION,
        framework_version=PYTORCH_LATEST_FRAMEWORK,
        **estimator_kwargs,
    )


def huggingface_estimator(**estimator_kwargs) -> HuggingFace:
    """
    Get the Huggingface sagemaker estimator with the most up-to-date framework version.
    List of available containers: https://github.com/aws/deep-learning-containers/blob/master/available_images.md

    :param estimator_kwargs: Estimator parameters as discussed in
        https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html
    :return: PyTorch estimator
    """
    return HuggingFace(
        framework_version=HUGGINGFACE_LATEST_FRAMEWORK_VERSION,
        transformers_version=HUGGINGFACE_LATEST_TRANSFORMERS_VERSION,
        pytorch_version=HUGGINGFACE_LATEST_PYTORCH_VERSION,
        py_version=HUGGINGFACE_LATEST_PY_VERSION,
        **estimator_kwargs,
    )


def sklearn_estimator(**estimator_kwargs) -> SKLearn:
    """
    Get the Scikit-learn sagemaker estimator with the most up-to-date framework version.
    List of available containers: https://github.com/aws/deep-learning-containers/blob/master/available_images.md

    :param estimator_kwargs: Estimator parameters as discussed in
        https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html
    :return: PyTorch estimator
    """
    return SKLearn(
        framework_version="1.0-1",
        py_version="py3",
        **estimator_kwargs,
    )


def mxnet_estimator(**estimator_kwargs) -> MXNet:
    """
    Get the MXNet sagemaker estimator with the most up-to-date framework version.
    List of available containers: https://github.com/aws/deep-learning-containers/blob/master/available_images.md

    :param estimator_kwargs: Estimator parameters as discussed in
        https://sagemaker.readthedocs.io/en/stable/frameworks/mxnet/sagemaker.mxnet.html
    :return: PyTorch estimator
    """
    return MXNet(
        framework_version=MXNET_LATEST_VERSION,
        py_version=MXNET_LATEST_PY_VERSION,
        **estimator_kwargs,
    )


sagemaker_estimator = {
    "PyTorch": pytorch_estimator,
    "HuggingFace": huggingface_estimator,
    "BasicCPU": basic_cpu_instance_sagemaker_estimator,
    "MXNet": MXNet,
    "TensorFlow": TensorFlow,
    "SKLearn": sklearn_estimator,
    "Chainer": Chainer,
    "XGBoost": XGBoost,
}
