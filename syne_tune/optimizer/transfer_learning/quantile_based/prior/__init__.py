from typing import Tuple

import numpy as np


class FunctionalPrior:
    def __init__(
            self,
            # todo is may be better to pass tensor as arguments and unify whether we use tensor/np array
            X_train: np.array,
            y_train: np.array,
    ):
        super(FunctionalPrior, self).__init__()
        assert len(X_train) == len(y_train)
        assert X_train.ndim == 2
        assert y_train.ndim == 2
        self.dim = X_train.shape[1]

    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        """
        :param X: features with shape (n, dim)
        :return: two arrays with shape (n,)
        """
        pass