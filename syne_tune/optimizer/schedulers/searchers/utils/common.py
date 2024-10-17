from typing import Union, Dict, Callable


Hyperparameter = Union[str, int, float]

Configuration = Dict[str, Hyperparameter]

# Type of ``filter_observed_data``, which is (optionally) used to filter the
# observed data in ``TuningJobState.trials_evaluations`` when determining
# the best config (incumbent) or the exclusion list. One use case is
# warm-starting, where the observed data can come from a number of tasks, only
# one of which is active.

ConfigurationFilter = Callable[[Configuration], bool]
