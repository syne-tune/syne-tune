from functools import partial

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    AcquisitionFunctionConstructor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
    LCBAcquisitionFunction,
)


SUPPORTED_ACQUISITION_FUNCTIONS = (
    "ei",
    "lcb",
)


def acquisition_function_factory(name: str, **kwargs) -> AcquisitionFunctionConstructor:
    assert (
        name in SUPPORTED_ACQUISITION_FUNCTIONS
    ), f"name = {name} not supported. Choose from:\n{SUPPORTED_ACQUISITION_FUNCTIONS}"
    if name == "ei":
        return EIAcquisitionFunction
    else:  # name == "lcb"
        kappa = kwargs.get("kappa", 1.0)
        return partial(LCBAcquisitionFunction, kappa=kappa)
