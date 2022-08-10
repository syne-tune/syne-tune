import numpy as np
import argparse
import pickle
import os
import time
import torch
import pandas as pd 
import scipy as sp
import torch.nn as nn
import torch.optim as optim


if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    def forward(self, x1, b):
        f1 = self.activate(self.fc1(x1))
        f2 = self.activate(self.fc2(b * f1))
        return self.fc3(f2)
    
    
class Exploitation:
    def __init__(self, dim, lr = 0.001, hidden=100):
        '''dim: number of dimensions of input'''    
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''
        
        self.func = Network_exploitation(dim, hidden_size=hidden).to(device)
        self.x1_list = []
        self.b_list = []
        self.reward = []
        self.lr = lr
        self.brackets = 1
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        
        self.data_size = 0
        self.sum_b = 0.01
        self.average_b = 0.01
        self.max_b = 0.01
    
    
    def add_data(self, x, reward):
        x1 = torch.tensor(x[0]).float()
        b = torch.tensor(x[1]).float()
        self.x1_list.append(x1)
        self.b_list.append(b) 
        self.reward.append(reward)
        self.data_size +=1
        self.sum_b += x[1]
        if self.max_b < x[1]:
            self.max_b = x[1]
        self.average_b = float(self.sum_b/self.data_size)
        #if self.data_size% 500 == 0:
         #   print("data points:", self.data_size)
        
        
    def predict(self, x):
        x1 = torch.tensor(x[0]).float().to(device)
        b = torch.tensor(x[1]).float().to(device)
        res = self.func(x1, b)
        return res

    
    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        length = len(self.reward)    
        index = np.arange(length)    
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
           # np.random.shuffle(index)
            for idx in index:
                x1 =self.x1_list[idx].to(device)
                b = self.b_list[idx].to(device)
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(x1, b) - r)**2 
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 500:
                    #print("Force Exploitation Net Loss", tot_loss / cnt)
                    #print("batch_loss / length", batch_loss / length)
                    return tot_loss / cnt
            if batch_loss / length <= 1e-4:
                #print("Exploitation Net Loss", batch_loss / length)
                return batch_loss / length

    
    

               