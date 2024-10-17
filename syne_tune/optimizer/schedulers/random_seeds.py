import numpy as np
from numpy.random import RandomState


RANDOM_SEED_UPPER_BOUND = 2**31 - 1


def generate_random_seed(random_state: RandomState = np.random) -> int:
    return random_state.randint(0, RANDOM_SEED_UPPER_BOUND)


class RandomSeedGenerator:
    def __init__(self, master_seed: int):
        self._random_state = np.random.RandomState(master_seed)

    def __call__(self) -> int:
        return generate_random_seed(self._random_state)
