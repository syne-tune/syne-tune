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
"""
Collects constants to be shared between core code and tuning scripts or
benchmarks.
"""

SYNE_TUNE_ENV_FOLDER = "SYNETUNE_FOLDER"  # environment variable that allows to overides default library folder
SYNE_TUNE_DEFAULT_FOLDER = "syne-tune"  # name of default library folder used if the env variable is not defined

ST_TUNER_CREATION_TIMESTAMP = "st_tuner_creation_timestamp"
ST_TUNER_START_TIMESTAMP = "st_tuner_start_timestamp"

# constants of keys that are written by `report`
ST_WORKER_ITER = "st_worker_iter"
ST_WORKER_TIMESTAMP = "st_worker_timestamp"
ST_WORKER_TIME = "st_worker_time"
ST_WORKER_COST = "st_worker_cost"
ST_INSTANCE_TYPE = "st_instance_type"
ST_INSTANCE_COUNT = "st_instance_count"

# constants for tuner results
ST_TRIAL_ID = "trial_id"
ST_TUNER_TIMESTAMP = "st_tuner_timestamp"
ST_TUNER_TIME = "st_tuner_time"
ST_DECISION = "st_decision"
ST_STATUS = "st_status"

# constant for the hyperparameter name that contains the checkpoint directory
ST_CHECKPOINT_DIR = "st_checkpoint_dir"

# Name for `upload_dir` in `RemoteTuner`
ST_REMOTE_UPLOAD_DIR_NAME = "tuner"
