from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import matplotlib.pyplot as plt

#gpu-server setting
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#dataset
from sklearn import datasets

n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
x, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

x_data = torch.Tensor(x)
y_data = torch.Tensor(y.reshape(100, 1))

print(x_data[:5])
print(y_data[:5])

#graph set
def scatter_plot():
    plt.scatter(x[y==0, 0], x[y==0, 1])
    plt.scatter(x[y==1, 0], x[y==1, 1])

#scatter_plot()

#model set
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred

    def predict(selfself, x):
        return 1 if pred >= 0.5 else 0


torch.manual_seed(2)
net = Net(2, 1)
list(net.parameters())

#랜덤으로 가중치 2개와 편향을 생성
[w, b] = net.parameters()
w1, w2 = w.view(2)
b1 = b[0]

def get_params():
    return w1.item(), w2.item(), b.item()

print(w1.item(), w2.item(), b1.item())


