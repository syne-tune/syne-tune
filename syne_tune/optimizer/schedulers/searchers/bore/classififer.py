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

import GPy

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


class GPModel:

    def __init__(self, kernel_type: str = 'matern52'):
        self.kernel_type = kernel_type

    def fit(self, X, y):
        noise_prior = GPy.priors.Gamma(0.1, 0.1)
        noise_kernel = GPy.kern.White(X.shape[1])
        noise_kernel.set_prior(noise_prior)

        if self.kernel_type == 'matern52':
            kern = GPy.kern.Matern52(X.shape[1], ARD=True) + noise_kernel
        elif self.kernel_type == 'rbf':
            kern = GPy.kern.RBF(X.shape[1], ARD=True) + noise_kernel

        self.m = GPy.models.GPClassification(X, y[:, None], kernel=kern)
        self.m.optimize()

    def predict_proba(self, X):
        m = self.m.predict(X)[0]
        return m

    def predict(self, X):
        l = np.round(self.m.predict(X)[0])
        return l


class MLP:

    def __init__(self, n_inputs, n_hidden: int = 32, epochs: int = 100, learning_rate: float = 1e-3,
                 train_from_scratch: bool = True, activation: str = 'relu'):

        if activation == 'relu':
            self.act_func = nn.ReLU
        elif activation == 'elu':
            self.act_func = nn.ELU
        elif activation == 'tanh':
            self.act_func = nn.Tanh

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.train_from_scratch = train_from_scratch

        self.model = self._init_network(n_inputs=self.n_inputs, n_hidden=self.n_hidden, act_func=self.act_func)

    def _init_network(self, n_inputs, n_hidden, act_func):
        return nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def fit(self, X, y):

        if self.train_from_scratch:
            self.model = self._init_network(n_inputs=self.n_inputs, n_hidden=self.n_hidden, act_func=self.act_func)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCELoss()

        tensor_x = torch.Tensor(X).float()
        tensor_y = torch.Tensor(y[:, None]).float()

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset)

        self.model.train()
        for epoch in range(self.epochs):

            for i, (images, labels) in enumerate(dataloader):
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        m = self.model(torch.tensor(X).float())
        return m.detach().numpy()

    def predict(self, X):
        l = np.round(self.predict_proba(X))
        return l
