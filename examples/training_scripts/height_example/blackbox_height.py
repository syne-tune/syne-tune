from typing import Any
import numpy as np

from syne_tune.blackbox_repository.blackbox import Blackbox, ObjectiveFunctionResult
from syne_tune.config_space import randint
from examples.training_scripts.height_example.train_height import (
    train_height,
    height_config_space,
    RESOURCE_ATTR,
    METRIC_ATTR,
)


class HeightExampleBlackbox(Blackbox):
    def __init__(
        self,
        max_steps: int = 100,
        sleep_time: float = 0.1,
        elapsed_time_attr: str = "elapsed_time",
    ):
        config_space = height_config_space(max_steps)
        fidelity_space = {RESOURCE_ATTR: randint(1, max_steps)}
        super().__init__(
            configuration_space=config_space,
            fidelity_space=fidelity_space,
            objectives_names=[METRIC_ATTR, elapsed_time_attr],
        )
        self._max_steps = max_steps
        self._sleep_time = sleep_time
        self.num_seeds = 1

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        assert seed is None or seed == 0, "Blackbox has one seed only"
        width, height = configuration["width"], configuration["height"]
        if fidelity is None:
            columns = (
                np.array(
                    [
                        train_height(step, width, height)
                        for step in range(self._max_steps)
                    ]
                ),
                self.fidelity_values * self._sleep_time,
            )
            return np.hstack([v.reshape((-1, 1)) for v in columns])
        else:
            assert len(fidelity) == 1, "Blackbox has single fidelity"
            epoch = int(next(iter(fidelity.values())))
            return dict(
                zip(
                    self.objectives_names,
                    [train_height(epoch - 1, width, height), epoch * self._sleep_time],
                )
            )

    @property
    def fidelity_values(self) -> np.array | None:
        return np.arange(1, self._max_steps + 1)
