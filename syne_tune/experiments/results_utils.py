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

from syne_tune.constants import (
    ST_METADATA_FILENAME,
    ST_RESULTS_DATAFRAME_FILENAME,
)
from syne_tune.util import experiment_path, s3_experiment_path


def sync_from_s3_command(experiment_name: str, s3_bucket: Optional[str] = None) -> str:
    s3_source_path = s3_experiment_path(
        s3_bucket=s3_bucket, experiment_name=experiment_name
    )
    target_path = str(experiment_path() / experiment_name)
    return (
        f'aws s3 sync {s3_source_path} {target_path} --exclude "*" '
        f'--include "*{ST_METADATA_FILENAME}" '
        f'--include "*{ST_RESULTS_DATAFRAME_FILENAME}"'
    )
