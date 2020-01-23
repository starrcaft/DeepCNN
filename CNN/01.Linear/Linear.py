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

x= torch.randn(100, 1) * 10
y = x + 3 * torch.randn(100, 1)

plt.plot(x.numpy(), y.numpy(), 'o')

plt.ylabel('y')
plt.xlabel('x')

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

#random seed
torch.manual_seed(1)
model = Net(1, 1)
print(model)

print(list(model.parameters()))

w, b = model.parameters()

def get_params():
    return w[0][0].item(), b[0].item()

def plot_fit(title):
    plt.title = title
    #wegith 과 bias
    w1, b1 = get_params()
    #-30~30
    x1 = np.array([-30, 30])
    #선형 추정선을 그려본다
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(x, y)
    plt.show()

plot_fit('initialModel')