#torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

#torchvision_import
import torchvision
import torchvision.datasets as datasets
import torchvision.utils as utils
import torchvision.transforms as transforms

from matplotlib import cm

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import time

#gpu-server setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#set Cuda
is_cuda = torch.cuda.is_available()
device = torch.device(0 if is_cuda else "cpu")

#데이터는 그대로 사용해도 괜찮으나 신경망에 입력하기 위해 표준화하는것이 좋다
standardizator = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)#3개는 rgb를 위함, 하지만 본 데이터는 gray임
    )
])

#get Dataset from MNIST
train_data = datasets.MNIST(
    root='data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

#show img
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#batch를 위해 data_loader 구현

#60000
#print(len(train_data))

# 60000 / 200 = 300
batch_size = 200

train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size, shuffle=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size, shuffle=True
)

#N-Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #CNN Sequential
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ).to(device)
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        ).to(device)

    #Forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc(x)
        return x

#CNN Model
net = Net().to(device)

#Run Train and Test with CNN
def run_epoch(model, train_data, optimizer, criterion):
    star_time = time.time()

    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        imgs, labels = data

        #to device imgs and labels
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        #forward
        outputs = model.forward(imgs)

        #loss
        loss = criterion(outputs, labels)

        #back ward
        loss.backward()

        #optimizer for imgs
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    end_time = time.time()

    return end_time - star_time


#Optimizer(최적화) and Criterion(손실함수)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    response = run_epoch(net, train_data_loader, optimizer, criterion)
    print('response: ', response)

print('Finished Training')
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_data_loader)

images, labels = dataiter.next()

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))