from dataclasses import dataclass
from typing import Dict, Optional

from syne_tune.optimizer.schedulers.botorch.botorch_gp import BotorchGP
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.optimizer.transfer_learning.bounding_box import BoundingBox

@dataclass
class MethodArguments:
    config_space: Dict
    metric: str
    mode: str
    random_seed: int
    max_t: int
    resource_attr: str

    transfer_learning_evaluations: Optional[Dict] = None


methods = {
    'RS': lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    'HB': lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={'debug_log': False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    'RS-MSR': lambda method_arguments: MedianStoppingRule(
        scheduler=FIFOScheduler(
            config_space=method_arguments.config_space,
            searcher="random",
            metric=method_arguments.metric,
            mode=method_arguments.mode,
            random_seed=method_arguments.random_seed,
        ),
        resource_attr=method_arguments.resource_attr,
        running_average=False,
    ),
    'Botorch': lambda method_arguments: BotorchGP(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
    ),
    'GP': lambda method_arguments: FIFOScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options={'debug_log': False},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    'MOBSTER': lambda method_arguments: HyperbandScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options={'debug_log': False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    'RS-BB': lambda method_arguments: BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
            new_config_space,
            searcher='random',
            metric=metric,
            mode=mode,
        ),
        mode="min",
        metric=method_arguments.metric,
        config_space=method_arguments.config_space,
        transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
    ),
    'HB-BB': lambda method_arguments: BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: HyperbandScheduler(
            new_config_space,
            searcher='random',
            metric=metric,
            mode=mode,
            search_options={'debug_log': False},
            max_t=method_arguments.max_t,
            resource_attr=method_arguments.resource_attr,
            random_seed=method_arguments.random_seed,
        ),
        mode="min",
        metric=method_arguments.metric,
        config_space=method_arguments.config_space,
        transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
    )

}
