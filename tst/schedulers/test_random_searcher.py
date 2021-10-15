from sagemaker_tune.optimizer.schedulers.searchers.searcher import \
    RandomSearcher
from sagemaker_tune.search_space import choice, randint


def test_no_duplicates():
    config_spaces = [
        {'cat_attr': choice(['a', 'b'])},
        {'int_attr': randint(lower=0, upper=1)},
    ]
    num_suggest_to_fail = 3

    for config_space in config_spaces:
        searcher = RandomSearcher(config_space, metric='accuracy')
        for trial_id in range(num_suggest_to_fail):
            # These should not fail
            config = searcher.get_config(trial_id=trial_id)
            if trial_id < num_suggest_to_fail - 1:
                assert config is not None
            else:
                assert config is None
