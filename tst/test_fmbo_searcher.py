import numpy as np
import pytest

from syne_tune.config_space import randint, uniform

from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import (
    FMBOSearcher,
)


@pytest.mark.timeout(300)
def test_fmbo_searcher():

    config_space = {"b": randint(1, 100), "c": uniform(0, 1)}

    searcher = FMBOSearcher(
        config_space=config_space,
        checkpoint_dir="synetune/qwen3_2M_token_1B_lr_1e-2_bsz_8_seed_0",
        use_vllm=False,
    )

    for i in range(5):
        config = searcher.suggest()
        assert config_space["b"].lower <= config["b"] <= config_space["b"].upper
        assert config_space["c"].lower <= config["c"] <= config_space["c"].upper
        searcher.on_trial_complete(i, config, metric=np.random.rand())
