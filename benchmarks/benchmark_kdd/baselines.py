from syne_tune.optimizer.schedulers.botorch.botorch_gp import BotorchGP
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.optimizer.transfer_learning.bounding_box import BoundingBox

methods = {
    'RS': lambda config_space, metric, mode, random_seed, max_t, resource_attr: FIFOScheduler(
        config_space=config_space,
        searcher="random",
        metric=metric,
        mode=mode,
        random_seed=random_seed,
    ),
    'HB': lambda config_space, metric, mode, random_seed, max_t, resource_attr: HyperbandScheduler(
        config_space=config_space,
        searcher="random",
        search_options={'debug_log': False},
        mode=mode,
        metric=metric,
        max_t=max_t,
        resource_attr=resource_attr,
        random_seed=random_seed,
    ),
    'RS-MSR': lambda config_space, metric, mode, random_seed, max_t, resource_attr: MedianStoppingRule(
        scheduler=FIFOScheduler(
            config_space=config_space,
            searcher="random",
            metric=metric,
            mode=mode,
            random_seed=random_seed,
        ),
        resource_attr=resource_attr,
        running_average=False,
    ),
    'Botorch': lambda config_space, metric, mode, random_seed, max_t, resource_attr: BotorchGP(
        config_space=config_space,
        metric=metric,
        mode=mode,
    ),
    'GP': lambda config_space, metric, mode, random_seed, max_t, resource_attr: FIFOScheduler(
        config_space,
        searcher="bayesopt",
        search_options={'debug_log': False},
        metric=metric,
        mode=mode,
        random_seed=random_seed,
    ),
    'MOBSTER': lambda config_space, metric, mode, random_seed, max_t, resource_attr: HyperbandScheduler(
        config_space,
        searcher="bayesopt",
        search_options={'debug_log': False},
        mode=mode,
        metric=metric,
        max_t=max_t,
        resource_attr=resource_attr,
        random_seed=random_seed,
    ),
    'HB+BB': lambda config_space, metric, mode, random_seed, max_t, resource_attr: BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
            new_config_space,
            searcher='random',
            metric=metric,
            mode=mode,
        ),
        mode="min",
        metric=metric,
        config_space=config_space,
        transfer_learning_evaluations=transfer_learning_evaluations,
    )
}
