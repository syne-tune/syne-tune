"""
Collects constants to be shared between core code and tuning scripts or
benchmarks.
"""

SYNE_TUNE_ENV_FOLDER = "SYNETUNE_FOLDER"
"""Environment variable that allows to overides default library folder"""  # pylint: disable=W0105

SYNE_TUNE_DEFAULT_FOLDER = "syne-tune"
"""Name of default library folder used if the env variable is not defined"""  # pylint: disable=W0105

ST_TUNER_CREATION_TIMESTAMP = "st_tuner_creation_timestamp"

ST_TUNER_START_TIMESTAMP = "st_tuner_start_timestamp"


# Constants of keys that are written by ``report``

ST_WORKER_ITER = "st_worker_iter"
"""Number of times reporter was called"""  # pylint: disable=W0105

ST_WORKER_TIMESTAMP = "st_worker_timestamp"
"""Time stamp when worker was called"""  # pylint: disable=W0105

ST_WORKER_TIME = "st_worker_time"
"""Time since creation of reporter"""  # pylint: disable=W0105

ST_WORKER_COST = "st_worker_cost"
"""Estimate of dollar cost spent so far"""  # pylint: disable=W0105

# Constants for tuner results

ST_TRIAL_ID = "trial_id"

ST_TUNER_TIMESTAMP = "st_tuner_timestamp"

ST_TUNER_TIME = "st_tuner_time"

ST_DECISION = "st_decision"

ST_STATUS = "st_status"

ST_CHECKPOINT_DIR = "st_checkpoint_dir"
"""Name of config key for checkpoint directory"""  # pylint: disable=W0105

ST_CONFIG_JSON_FNAME_ARG = "st_config_json_filename"
"""Name of config key for config JSON file"""  # pylint: disable=W0105

ST_REMOTE_UPLOAD_DIR_NAME = "tuner"
"""Name for ``upload_dir`` in ``RemoteTuner``"""  # pylint: disable=W0105


# File names

ST_RESULTS_DATAFRAME_FILENAME = "results.csv.zip"
"""Name for results dataframe stored in ``StoreResultsCallback``"""  # pylint: disable=W0105

ST_METADATA_FILENAME = "metadata.json"
"""Name for metadata file stored in ``Tuner``"""  # pylint: disable=W0105

ST_TUNER_DILL_FILENAME = "tuner.dill"
"""Name for final tuner object file stored in ``Tuner``"""  # pylint: disable=W0105

ST_DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
"""Datetime format used in result path names"""  # pylint: disable=W0105

TUNER_DEFAULT_SLEEP_TIME = 5.0
"""Default value for ``sleep_time``"""  # pylint: disable=W0105

ST_METRIC_TAG = "tune-metric"
"""Tag for log lines used in :class:`~syne_tune.Reporter`"""  # pylint: disable=W0105
