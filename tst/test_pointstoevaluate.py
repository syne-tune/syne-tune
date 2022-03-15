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
from typing import Dict
import numpy as np

from syne_tune.config_space import randint, lograndint, uniform, \
    loguniform, choice, finrange, logfinrange
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers import RandomSearcher


def _to_int(a, lower, upper):
    return int(np.clip(round(a), lower, upper))


def _to_float(a, lower, upper):
    return float(np.clip(a, lower, upper))


def _lin_avg(a, b):
    return 0.5 * (b + a)


def _log_avg(a, b):
    return np.exp(0.5 * (np.log(b) + np.log(a)))


def _impute_config(config: Dict) -> Dict:
    new_config = config.copy()
    k, lower, upper = 'int', 1, 5
    if k not in config:
        new_config[k] = _to_int(_lin_avg(lower, upper), lower, upper)
    k, lower, upper = 'logint', 3, 15
    if k not in config:
        new_config[k] = _to_int(_log_avg(lower, upper), lower, upper)
    k, lower, upper = 'float', 5.5, 6.5
    if k not in config:
        new_config[k] = _to_float(_lin_avg(lower, upper), lower, upper)
    k, lower, upper = 'logfloat', 7.5, 8.5
    if k not in config:
        new_config[k] = _to_float(_log_avg(lower, upper), lower, upper)
    k = 'categorical'
    if k not in config:
        new_config[k] = 'a'
    k = 'finrange'
    if k not in config:
        new_config[k] = 0.5
    k = 'logfinrange'
    if k not in config:
        new_config[k] = np.exp(3.0)
    return new_config


def _gen_testcase(inds, config_pairs):
    a, b = zip(*[config_pairs[i] for i in inds])
    return list(a), list(b)


def _remove_duplicates(inds):
    excl_set = set()
    result = []
    for i in inds:
        if i not in excl_set:
            result.append(i)
            excl_set.add(i)
    return result


def _prepare_for_compare(configs1, configs2, hp_ranges):
    res1, res2 = [], []
    for c1, c2 in zip(configs1, configs2):
        _c1 = hp_ranges.config_to_tuple(c1)
        _c2 = hp_ranges.config_to_tuple(c2)

        def remap(a, b):
            if isinstance(a, str):
                return (1, int(a == b))
            else:
                return (a, b)

        t1, t2 = zip(*[remap(a, b) for a, b in zip(_c1, _c2)])
        res1.extend(t1)
        res2.extend(t2)
    return res1, res2


def _prepare_test(is_ray_tune=False):
    np.random.seed(2838748673)
    num_extra_cases = 30

    config_space = {
        'int': randint(1, 5),
        'logint': lograndint(3, 15),
        'float': uniform(5.5, 6.5),
        'logfloat': loguniform(7.5, 8.5),
        'categorical': choice(['a', 'b', 'c']),
    }
    configs = [
        dict(),
        {'float': 5.75, 'categorical': 'b'},
        {'int': 1, 'logint': 3, 'float': 5.5, 'logfloat': 7.5, 'categorical': 'a'},
        {'int': 5, 'logint': 15, 'float': 6.5, 'logfloat': 8.5, 'categorical': 'c'},
        {'logfloat': 8.125, 'logint': 5, 'int': 4},
        {'float': 6.125},
        {'categorical': 'c'},
    ]
    if not is_ray_tune:
        config_space.update({
            'finrange': finrange(0.1, 0.9, 9),
            'logfinrange': logfinrange(1.0, np.exp(6.0), 7),
        })
        configs.extend([
            {'finrange': 0.3},
            {'logfinrange': np.exp(2.0)},
        ])

    num_configs = len(configs)
    config_pairs = [(c, _impute_config(c)) for c in configs]
    #for a, b in config_pairs:
    #    print(f"{a} --- {b}")
    testcases = [
        (None, [config_pairs[0][1]]),
        ([], [])] + [_gen_testcase([i], config_pairs)
                     for i in range(num_configs)]
    testcases.append(
        (_gen_testcase([0, 2, 4, 1, 4, 2], config_pairs)[0],
         _gen_testcase([0, 2, 4, 1], config_pairs)[1]))
    for i in range(num_extra_cases):
        size = np.random.randint(1, 10)
        inds = np.random.randint(0, num_configs, size=size)
        tc_src, _ = _gen_testcase(inds, config_pairs)
        _, tc_trg = _gen_testcase(_remove_duplicates(inds), config_pairs)
        testcases.append((tc_src, tc_trg))

    return config_space, configs, testcases


def test_points_to_evaluate():
    config_space, configs, testcases = _prepare_test()
    search_options = {'debug_log': False}
    for tc_src, tc_trg in testcases:
        err_msg = f"tc_src = {tc_src}\ntc_trg = {tc_trg}"
        scheduler = FIFOScheduler(
            config_space, searcher='random', search_options=search_options,
            mode='min', metric='bogus', points_to_evaluate=tc_src)
        searcher: RandomSearcher = scheduler.searcher
        hp_ranges = searcher._hp_ranges
        assert len(tc_trg) == len(searcher._points_to_evaluate), err_msg
        assert np.allclose(*_prepare_for_compare(
            tc_trg, searcher._points_to_evaluate, hp_ranges)), err_msg
        tc_cmp = [scheduler.suggest(trial_id=i).config for i in range(len(tc_trg))]
        assert len(tc_trg) == len(tc_cmp), err_msg
        assert np.allclose(*_prepare_for_compare(
            tc_trg, tc_cmp, hp_ranges)), err_msg


def test_points_to_evaluate_raytune():
    from ray.tune.schedulers import FIFOScheduler as RT_FIFOScheduler
    from syne_tune.optimizer.schedulers.ray_scheduler import \
        RayTuneScheduler
    from syne_tune.optimizer.schedulers.searchers import \
        impute_points_to_evaluate

    config_space, configs, testcases = _prepare_test(is_ray_tune=True)
    # This is just to get hp_ranges, which is needed for comparisons below
    _myscheduler = FIFOScheduler(
        config_space, searcher='random', mode='min', metric='bogus')
    _mysearcher: RandomSearcher = _myscheduler.searcher
    hp_ranges = _mysearcher._hp_ranges
    for tc_src, tc_trg in testcases:
        err_msg = f"tc_src = {tc_src}\ntc_trg = {tc_trg}"
        ray_scheduler = RT_FIFOScheduler()
        ray_scheduler.set_search_properties(mode='min', metric='bogus')
        scheduler = RayTuneScheduler(
            config_space=config_space,
            ray_scheduler=ray_scheduler,
            points_to_evaluate=impute_points_to_evaluate(tc_src, config_space))
        tc_cmp = [scheduler.suggest(trial_id=i).config for i in range(len(tc_trg))]
        assert len(tc_trg) == len(tc_cmp), err_msg
        assert np.allclose(*_prepare_for_compare(
            tc_trg, tc_cmp, hp_ranges)), err_msg
