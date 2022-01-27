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
from benchmarking.cli.estimator_factory import sagemaker_estimator_factory
from benchmarking.cli.benchmark_factory import benchmark_factory, \
    supported_benchmarks
from syne_tune.backend.sagemaker_backend.sagemaker_utils import \
    get_execution_role
from syne_tune.util import repository_root_path


def test_create_estimators():
    try:
        role = get_execution_role()
        for benchmark_name in supported_benchmarks():
            benchmark = benchmark_factory({'benchmark_name': benchmark_name})
            def_params = benchmark['default_params']
            framework = def_params.get('framework')
            if framework is not None:
                sm_estimator = sagemaker_estimator_factory(
                    entry_point=benchmark['script'],
                    instance_type=def_params['instance_type'],
                    framework=framework,
                    role=role,
                    dependencies=[str(repository_root_path() / "benchmarking")],
                    framework_version=def_params.get('framework_version'),
                    pytorch_version=def_params.get('pytorch_version'))
    except Exception:
        print("Cannot run this test, because SageMaker role is not specified, "
              "and it cannot be inferred")
