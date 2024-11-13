import numpy as np
from numpy.random import RandomState

from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        activation: str = "relu",
        random_state: RandomState = None,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = MLPClassifier(
            activation=activation,
            hidden_layer_sizes=(n_hidden,),
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return np.round(self.predict_proba(X))
