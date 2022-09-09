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
import numpy as np

from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        activation: str = "relu",
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = MLPClassifier(
            activation=activation, hidden_layer_sizes=(n_hidden,)
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return np.round(self.predict_proba(X))
