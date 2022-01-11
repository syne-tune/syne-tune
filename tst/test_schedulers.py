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
import pytest
import itertools

from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband_impl import \
    SynchronousGeometricHyperbandScheduler
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.search_space import randint
from syne_tune.util import script_checkpoint_example_path
from tst.util_test import temporary_local_backend

_async_parameterizations = list(itertools.product(
    ['fifo', 'hyperband_stopping', 'hyperband_promotion'],
    ['random', 'bayesopt'],
    ['min', 'max']))

@pytest.mark.parametrize(
    "scheduler, searcher, mode", _async_parameterizations)
def test_async_scheduler(scheduler, searcher, mode):
    max_steps = 5
    num_workers = 2
    random_seed = 382378624

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        "sleep_time": 0.001
    }

    entry_point = script_checkpoint_example_path()
    metric = 'mean_loss'

    backend = temporary_local_backend(entry_point=entry_point)

    search_options = {
        'debug_log': False,
        'num_init_random': num_workers}

    if scheduler == 'fifo':
        myscheduler = FIFOScheduler(
            config_space,
            searcher=searcher,
            search_options=search_options,
            mode=mode,
            metric=metric,
            random_seed=random_seed)
    else:
        prefix = 'hyperband_'
        assert scheduler.startswith(prefix)
        sch_type = scheduler[len(prefix):]
        myscheduler = HyperbandScheduler(
            config_space,
            searcher=searcher,
            search_options=search_options,
            max_t=max_steps,
            type=sch_type,
            resource_attr='epoch',
            random_seed=random_seed,
            mode=mode,
            metric=metric)

    stop_criterion = StoppingCriterion(max_wallclock_time=0.2)
    tuner = Tuner(
        backend=backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()


_sync_parameterizations = list(itertools.product(
    ['random', 'bayesopt'],
    ['min', 'max']))

@pytest.mark.parametrize(
    "searcher, mode", _sync_parameterizations)
def test_sync_scheduler(searcher, mode):
    max_steps = 5
    num_workers = 2
    batch_size = 5
    random_seed = 382378624

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        "sleep_time": 0.001
    }

    entry_point = script_checkpoint_example_path()
    metric = 'mean_loss'

    backend = temporary_local_backend(entry_point=entry_point)

    search_options = {
        'debug_log': False,
        'num_init_random': num_workers,
        'num_init_candidates_for_batch': 100}

    myscheduler = SynchronousGeometricHyperbandScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode=mode,
        metric=metric,
        resource_attr='epoch',
        max_resource_attr='steps',
        random_seed=random_seed,
        batch_size=batch_size)

    stop_criterion = StoppingCriterion(max_wallclock_time=0.2)
    tuner = Tuner(
        backend=backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()


def test_hyperband_max_t_inference():
    config_space1 = {
        'epochs': 15,
        'max_t': 14,
        'max_epochs': 13,
        'blurb': randint(0, 20)
    }
    config_space2 = {
        'max_t': 14,
        'max_epochs': 13,
        'blurb': randint(0, 20)
    }
    config_space3 = {
        'max_epochs': 13,
        'blurb': randint(0, 20)
    }
    config_space4 = {
        'epochs': randint(15, 20),
        'max_t': 14,
        'max_epochs': 13,
        'blurb': randint(0, 20)
    }
    config_space5 = {
        'epochs': randint(15, 20),
        'max_t': randint(14, 21),
        'max_epochs': 13,
        'blurb': randint(0, 20)
    }
    config_space6 = {
        'blurb': randint(0, 20)
    }
    config_space7 = {
        'epochs': randint(15, 20),
        'max_t': randint(14, 21),
        'max_epochs': randint(13, 22),
        'blurb': randint(0, 20)
    }
    # Fields: (max_t, search_space, final_max_t)
    # If final_max_t is None, an assertion should be raised
    cases = [
        (None, config_space1, 15),
        (None, config_space2, 14),
        (None, config_space3, 13),
        (None, config_space4, 14),
        (None, config_space5, 13),
        (None, config_space6, None),
        (None, config_space7, None),
        (10, config_space1, 10),
        (10, config_space2, 10),
        (10, config_space3, 10),
        (10, config_space4, 10),
        (10, config_space5, 10),
        (10, config_space6, 10),
        (10, config_space7, 10),
    ]

    for max_t, configspace, final_max_t in cases:
        if final_max_t is not None:
            myscheduler = HyperbandScheduler(
                configspace,
                searcher='random',
                max_t=max_t,
                resource_attr='epoch',
                mode='max',
                metric='accuracy')
            assert final_max_t == myscheduler.max_t
        else:
            with pytest.raises(AssertionError):
                myscheduler = HyperbandScheduler(
                    configspace,
                    searcher='random',
                    max_t=max_t,
                    resource_attr='epoch',
                    mode='max',
                    metric='accuracy')
