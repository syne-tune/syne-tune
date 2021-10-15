from datetime import datetime

from sagemaker_tune.backend.trial_status import Trial
from sagemaker_tune.search_space import randint
from sagemaker_tune.optimizer.schedulers.pbt import PopulationBasedTraining

max_steps = 10

config_space = {
    "width": randint(0, 20),
}
resource_attr = "step"
metric = "mean_loss"

total_steps = 10
population_size = 2

pbt = PopulationBasedTraining(config_space=config_space,
                              metric=metric,
                              resource_attr=resource_attr,
                              population_size=population_size,
                              mode='min',
                              max_t=total_steps,
                              perturbation_interval=1)


def update_state(suggest, state):
    """
    Mock-up of a backend to simulate the backend state after suggesting a new trial
    """

    if suggest.spawn_new_trial_id:
        i = len(state.keys())
        t = Trial(config=suggest.config, trial_id=i, creation_time=datetime.now())
        state[t.trial_id] = {}
        state[t.trial_id]['trial'] = t
        state[t.trial_id]['step'] = 1
        trial_id = t.trial_id
    else:
        trial_id, config = suggest.config
        t = state[trial_id]['trial']
        t.config = config
        state[trial_id]['trial'] = t
        state[trial_id]['step'] += 1

    return trial_id


def test_ptb():

    state = {}
    for i in range(total_steps):
        suggest = pbt.suggest(i)

        if i < population_size:
            # first configs should be random
            assert suggest.spawn_new_trial_id
            assert suggest.checkpoint_trial_id is None

        else:
            # make sure that we keep config 0 in the population since it's the best
            assert suggest.checkpoint_trial_id == 0

            # do we have a new config
            assert suggest.config != state[0]['trial'].config

        trial_id = update_state(suggest, state)
        t = state[trial_id]['trial']
        pbt.on_trial_add(t)
        results = {metric: i, resource_attr: state[trial_id]['step']}

        s = pbt.on_trial_result(t, results)

        if i > 1:
            # all trials after the first one should be stopped and resampled
            assert s == 'PAUSE'

    # add better config
    trial_id = update_state(suggest, state)
    t = state[trial_id]['trial']
    pbt.on_trial_add(t)
    results = {metric: -1, resource_attr: state[trial_id]['step']}

    s = pbt.on_trial_result(t, results)
    assert s == 'CONTINUE'

    # config 0's performance dropped
    t = state[0]['trial']
    results = {metric: 100, resource_attr: state[trial_id]['step'] + 1}
    s = pbt.on_trial_result(t, results)
    assert s == 'PAUSE'

    # we should now continue with config 10
    suggest = pbt.suggest(total_steps)
    assert suggest.checkpoint_trial_id == 10
