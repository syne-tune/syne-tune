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
import pytest

import benchmarking
from benchmarking.commons.benchmark_definitions import real_benchmark_definitions
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from syne_tune.remote.estimators import sagemaker_estimator
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    default_sagemaker_session,
)

all_real_benchmarks = [
    (bm, False) for bm in real_benchmark_definitions(sagemaker_backend=False).values()
] + [(bm, True) for bm in real_benchmark_definitions(sagemaker_backend=True).values()]

try:
    sm_session = default_sagemaker_session()
except Exception:
    print(
        "Cannot run this test, because SageMaker role is not specified, "
        "and it cannot be inferred"
    )
    sm_session = None


@pytest.mark.parametrize("benchmark, sagemaker_backend", all_real_benchmarks)
def test_create_estimators(benchmark, sagemaker_backend):
    if sm_session is not None:
        sm_args = sagemaker_estimator_args(
            entry_point=benchmark.script,
            experiment_tag="A",
            tuner_name="B",
            benchmark=benchmark,
            sagemaker_backend=False,
        )
        sm_args["sagemaker_session"] = sm_session
        sm_args["dependencies"] = benchmarking.__path__
        sm_estimator = sagemaker_estimator[benchmark.framework](**sm_args)
