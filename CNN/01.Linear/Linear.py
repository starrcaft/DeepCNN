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

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#100회 반복
epoch = 100
losses = []
for i in range(epoch):
    #예측
    y_pred = model.forward(x)
    #오차계산
    loss = criterion(y_pred, y)
    print("epoch:", i, "loss:", loss.item())

    #오차 누적(계산을 위해서)
    losses.append(loss)
    #optimzer 초기화
    optimizer.zero_grad()
    #backward를 수행하여 그래디언트 계산
    loss.backward()
    #learning rate 만큼 가중치를 주어서 hyper parameter 업데이트
    optimizer.step()

#진행과 loss에 대한 그래프
#plt.plot(range(epoch), losses)
#plt.ylabel('Loss')
#plt.xlabel('epoch')

plot_fit("Trained Model")
plt.show()