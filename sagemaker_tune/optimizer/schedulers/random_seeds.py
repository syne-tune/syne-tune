import numpy as np


class RandomSeedGenerator(object):
    def __init__(self, master_seed: int):
        self._random_state = np.random.RandomState(master_seed)

    def __call__(self) -> int:
        return self._random_state.randint(0, 2 ** 32)
