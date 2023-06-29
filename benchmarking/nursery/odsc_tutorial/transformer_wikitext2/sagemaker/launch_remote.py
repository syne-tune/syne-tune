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
from pathlib import Path

from transformer_wikitext2.baselines import methods
from transformer_wikitext2.benchmark_definitions import benchmark_definitions
from syne_tune.experiments.launchers.launch_remote_sagemaker import launch_remote


if __name__ == "__main__":
    entry_point = Path(__file__).parent / "hpo_main.py"
    source_dependencies = [str(Path(__file__).parent.parent)]
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=source_dependencies,
    )
