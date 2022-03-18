from syne_tune.optimizer.schedulers import HyperbandScheduler, FIFOScheduler

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
}
