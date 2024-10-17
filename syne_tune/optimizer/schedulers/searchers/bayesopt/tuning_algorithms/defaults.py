from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    LBFGSOptimizeAcquisition,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
)


DEFAULT_ACQUISITION_FUNCTION = EIAcquisitionFunction

DEFAULT_LOCAL_OPTIMIZER_CLASS = LBFGSOptimizeAcquisition

DEFAULT_NUM_INITIAL_CANDIDATES = 250

DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS = 3

DEFAULT_MAX_SIZE_DATA_FOR_MODEL = 500
