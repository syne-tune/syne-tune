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
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class NetworkExploitation(nn.Module):
    def __init__(self, dim: int, hidden_size: int = 100):
        super(NetworkExploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x1, b):
        f1 = self.activate(self.fc1(x1))
        f2 = self.activate(self.fc2(b * f1))
        return self.fc3(f2)


class Exploitation:
    def __init__(self, dim: int, lr: float = 0.001, hidden: int = 100):

        """
        the budget-aware network of NeuralBand

        :param dim: number of dimensions of configuration vector
        :param lr: learning rate of Adam
        :param hidden: width of neural network
        """
        self.lr = lr
        self.func = NetworkExploitation(dim, hidden_size=hidden).to(device)

        # store all configuration vectors
        self.x1_list = []
        # store all budgets
        self.b_list = []
        # store all evaluated scores
        self.reward_list = []

        # number of parameters of neural network
        self.total_param = sum(
            p.numel() for p in self.func.parameters() if p.requires_grad
        )
        # size of stored data
        self.data_size = 0

        # sum of all budgets
        self.sum_b = 0.01
        # average over all budgets
        self.average_b = 0.01
        # the maximal budget occured so far
        self.max_b = 0.01

    def add_data(self, x: list, reward: float):
        x1 = torch.tensor(x[0]).float()
        b = torch.tensor(x[1]).float()
        self.x1_list.append(x1)
        self.b_list.append(b)
        self.reward_list.append(reward)
        self.data_size += 1
        self.sum_b += x[1]
        if self.max_b < x[1]:
            self.max_b = x[1]
        self.average_b = float(self.sum_b / self.data_size)

    def predict(self, x: list) -> torch.Tensor:
        x1 = torch.tensor(x[0]).float().to(device)
        b = torch.tensor(x[1]).float().to(device)
        res = self.func(x1, b)
        return res

    def train(self) -> float:
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        length = len(self.reward_list)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                x1 = self.x1_list[idx].to(device)
                b = self.b_list[idx].to(device)
                r = self.reward_list[idx]
                optimizer.zero_grad()
                loss = (self.func(x1, b) - r) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 500:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-4:
                return batch_loss / length
