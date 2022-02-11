# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband_impl import \
    SynchronousGeometricHyperbandScheduler
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.constants import ST_WORKER_TIME

from benchmarking.cli.launch_utils import make_searcher_and_scheduler
from benchmarking.utils import dict_get

__all__ = ['scheduler_factory',
           'supported_schedulers',
           ]


def _check_searcher(searcher, supported_searchers):
    assert searcher is not None, \
        "searcher needs to be provided"
    assert searcher in supported_searchers, \
        f"searcher = '{searcher}' not supported ({supported_searchers})"


supported_schedulers = {
    'fifo',
    'hyperband_stopping',
    'hyperband_promotion',
    'hyperband_cost_promotion',
    'hyperband_pasha',
    'hyperband_synchronous',
    'mo_asha',
    'raytune_fifo',
    'raytune_hyperband',
}


# Note: If schedulers are the same for async and sync, only the async
# names are listed here
schedulers_with_search_options = {
    'fifo': FIFOScheduler,
    'hyperband_stopping': HyperbandScheduler,
    'hyperband_promotion': HyperbandScheduler,
    'hyperband_cost_promotion': HyperbandScheduler,
    'hyperband_pasha': HyperbandScheduler,
    'hyperband_synchronous': SynchronousGeometricHyperbandScheduler,
}


def scheduler_factory(
        params: dict, benchmark: dict, default_params: dict) -> (
        TrialScheduler, dict):
    """
    Creates scheduler from command line parameters and benchmark descriptor.
    We also return the CL parameters extended by benchmark-specific default
    values.

    :param params: CL parameters
    :param benchmark: Benchmark descriptor
    :param default_params: Default params for benchmark
    :return: scheduler, imputed_params

    """
    params = params.copy()
    config_space = benchmark['config_space']

    scheduler = params['scheduler']
    assert scheduler in supported_schedulers, \
        f"scheduler = '{scheduler}' not supported ({supported_schedulers})"
    _default_params = dict(instance_type='ml.m4.xlarge', num_workers=4)
    _default_params.update(default_params)
    for k, v in _default_params.items():
        if params.get(k) is None:
            params[k] = v
    if params.get('searcher_num_init_random') is None:
        # The default value for this is num_workers + 2
        params['searcher_num_init_random'] = params['num_workers'] + 2

    if scheduler in schedulers_with_search_options:
        searcher = params.get('searcher')
        if searcher is None:
            searcher = 'random'
            params['searcher'] = searcher
        else:
            supported_searchers = {'random', 'bayesopt', 'kde'}
            if scheduler == 'fifo':
                supported_searchers.update(
                    {'bayesopt_cost_coarse', 'bayesopt_cost_fine',
                     'bayesopt_constrained'})
            elif scheduler != 'hyperband_synchronous':
                supported_searchers.add('bayesopt_cost')
            _check_searcher(searcher, supported_searchers)

        # Searcher and scheduler options from params
        search_options, scheduler_options = make_searcher_and_scheduler(params)
        for k in ('metric', 'mode', 'max_resource_attr'):
            if k in benchmark:
                scheduler_options[k] = benchmark[k]
        if scheduler.startswith('hyperband'):
            k = 'resource_attr'
            if k in benchmark:
                scheduler_options[k] = benchmark[k]
        if scheduler == 'hyperband_cost_promotion' or searcher.startswith(
                'bayesopt_cost'):
            # Benchmark may define 'cost_attr'. If not, check for
            # 'elapsed_time_attr'
            cost_attr = None
            keys = ('cost_attr', 'elapsed_time_attr')
            for k in keys:
                if k in benchmark:
                    cost_attr = benchmark[k]
                    break
            if cost_attr is not None:
                if scheduler.startswith('hyperband'):
                    scheduler_options['cost_attr'] = cost_attr
                if searcher.startswith('bayesopt_cost'):
                    search_options['cost_attr'] = cost_attr
        k = 'points_to_evaluate'
        if k in params:
            scheduler_options[k] = params.get(k)
        # Transfer benchmark -> search_options
        k = 'map_reward'
        if k in benchmark:
            search_options[k] = benchmark[k]
        if searcher == 'bayesopt_cost_fine' or searcher == 'bayesopt_cost':
            keys = ('cost_model', 'resource_attr')
        elif searcher == 'bayesopt_constrained':
            keys = ('constraint_attr',)
        else:
            keys = ()
        for k in keys:
            v = benchmark.get(k)
            assert v is not None, \
                f"searcher = '{searcher}': Need {k} to be defined for " +\
                "benchmark"
            search_options[k] = v
        if searcher.startswith('bayesopt_cost'):
            searcher = 'bayesopt_cost'  # Internal name
        if scheduler == 'hyperband_pasha':
            rung_system_kwargs = scheduler_options.get(
                'rung_system_kwargs', dict())
            for name, tp in (('ranking_criterion', str), ('epsilon', float),
                             ('epsilon_scaling', float)):
                name_cl = 'pasha_' + name
                v = params.get(name_cl)
                if v is not None:
                    rung_system_kwargs[name] = tp(v)
            if rung_system_kwargs:
                scheduler_options['rung_system_kwargs'] = rung_system_kwargs
        # Build scheduler and searcher
        scheduler_cls = schedulers_with_search_options[scheduler]
        myscheduler = scheduler_cls(
            config_space,
            searcher=searcher,
            search_options=search_options,
            **scheduler_options)
    elif scheduler == 'mo_asha':
        # Use the mode for the first metric as given in the benchmark and
        # minimize time
        mode = [benchmark['mode'], 'min']
        metrics = [benchmark['metric'], ST_WORKER_TIME]
        myscheduler = MOASHA(
            config_space,
            mode=mode,
            metrics=metrics,
            max_t=params['max_resource_level'],
            time_attr=benchmark['resource_attr'])
    else:
        from ray.tune.schedulers import AsyncHyperBandScheduler
        from ray.tune.schedulers import FIFOScheduler as RT_FIFOScheduler
        from ray.tune.suggest.skopt import SkOptSearch
        from syne_tune.optimizer.schedulers.ray_scheduler import \
            RayTuneScheduler
        from syne_tune.optimizer.schedulers.searchers import \
            impute_points_to_evaluate

        searcher = params.get('searcher')
        if searcher is None:
            searcher = 'random'
            params['searcher'] = searcher
        else:
            _check_searcher(searcher, {'random', 'bayesopt'})
        rt_searcher = None  # Defaults to random
        metric = benchmark['metric']
        mode = benchmark['mode']
        points_to_evaluate = impute_points_to_evaluate(
            params.get('points_to_evaluate'), config_space)
        if searcher == 'bayesopt':
            rt_searcher = SkOptSearch(points_to_evaluate=points_to_evaluate)
            points_to_evaluate = None
            rt_searcher.set_search_properties(
                mode=mode, metric=metric,
                config=RayTuneScheduler.convert_config_space(config_space))
        if scheduler == 'raytune_hyperband':
            rt_scheduler = AsyncHyperBandScheduler(
                max_t=params['max_resource_level'],
                grace_period=dict_get(params, 'grace_period', 1),
                reduction_factor=dict_get(params, 'reduction_factor', 3),
                brackets=dict_get(params, 'brackets', 1),
                time_attr=benchmark['resource_attr'],
                mode=mode,
                metric=metric)
        else:
            rt_scheduler = RT_FIFOScheduler()
            rt_scheduler.set_search_properties(metric=metric, mode=mode)
        myscheduler = RayTuneScheduler(
            config_space=config_space,
            ray_scheduler=rt_scheduler,
            ray_searcher=rt_searcher,
            points_to_evaluate=points_to_evaluate)

    return myscheduler, params