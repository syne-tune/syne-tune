# File to store specifications for backends
# metric, mode, active_task_str, uses_fidelity

BACKEND_DEFS = {
    "SimOpt": (
        "profit",
        "max",
        "time_idx",
        False,
    ),
    "YAHPO": (
        "auc",
        "max",
        "trainsize",
        True,
    ),
    "XGBoost": (
        "metric_error",
        "min",
        "data_size",
        False,
    ),
}
