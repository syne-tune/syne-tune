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

SYNE_TUNE_FOLDER = "syne-tune"

SMT_TUNER_CREATION_TIMESTAMP = "st_tuner_creation_timestamp"
SMT_TUNER_START_TIMESTAMP = "st_tuner_start_timestamp"

# constants of keys that are written by `report`
SMT_WORKER_ITER = "st_worker_iter"
SMT_WORKER_TIMESTAMP = "st_worker_timestamp"
SMT_WORKER_TIME = "st_worker_time"
SMT_WORKER_COST = "st_worker_cost"
SMT_INSTANCE_TYPE = "st_instance_type"
SMT_INSTANCE_COUNT = "st_instance_count"

# constants for tuner results
SMT_TRIAL_ID = "trial_id"
SMT_TUNER_TIMESTAMP = "st_tuner_timestamp"
SMT_TUNER_TIME = "st_tuner_time"
SMT_DECISION = "st_decision"
SMT_STATUS = "st_status"

# constant for the hyperparameter name that contains the checkpoint directory
SMT_CHECKPOINT_DIR = "st_checkpoint_dir"

# Name for `upload_dir` in `RemoteTuner`
SMT_REMOTE_UPLOAD_DIR_NAME = "tuner"
