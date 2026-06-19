from typing import Any

from syne_tune.blackbox_repository.blackbox import Blackbox, ObjectiveFunctionResult


class BlackboxArtificial(Blackbox):
    """
    Base class for artificial blackbox functions. These are defined by a formula and
    are cheap to evaluate. They are mainly used for testing and to prototype new
    algorithms.
    """

    def __init__(
        self,
        dimension: int,
        configuration_space: dict[str, Any],
        objectives_names: list[str] | None = None,
    ):
        if objectives_names is None:
            objectives_names = ["y"]
        super().__init__(
            configuration_space=configuration_space,
            objectives_names=objectives_names,
        )
        self.dimension = dimension

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        raise NotImplementedError
