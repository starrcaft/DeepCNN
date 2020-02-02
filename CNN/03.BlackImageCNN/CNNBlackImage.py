#torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

#torchvision_import
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
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#set Cuda
is_cuda = torch.cuda.is_available()
device = torch.device(0 if is_cuda else "cpu")

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

image, label = train_data[0]

#데이터 정규화 이전 가시화 함수
print('image')
print('------------')
print('shape of this image\t', image.shape)
print('7\'th row of this image\t:', image[0][6])

print('label')
print('--------------')
print('label: ', label)



#데이터는 그대로 사용해도 괜찮으나 신경망에 입력하기 위해 표준화하는것이 좋다
standardizator = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)#3개는 rgb를 위함, 하지만 본 데이터는 gray임
    )
])


plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('%i' % label)
plt.show()

#정규화 이후 가시화 함수
def imgshow(img):
    img = (img + 1) / 2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imgshow(image)

batch_size = 200
#batch를 위해 data_loader 구현
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size, shuffle=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size, shuffle=True
)

#get example of batch_img
example_mini_batch_img, example_mini_batch_label, label = next(iter(train_data_loader))
print(example_mini_batch_img.shape)
print(example_mini_batch_label)

#DNN CODE
#mlp = nn.Sequential(
#    nn.Linear(28 * 28, 256),
#    nn.LeakyReLU(0.1),
#    nn.Linear(256, 10),
#    nn.Softmax(dim=-1)
#).to(device)

#change code to class NET
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
     #DNN Sequential
     #   self.test = nn.Sequential(
     #       nn.Linear(28 * 28, 256),
     #       nn.LeakyReLU(0.1),
     #       nn.Linear(256, 10),
     #       nn.Softmax(dim=-1)
     #   ).to(device)

    #CNN Forward
    def c_forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc(x)
        return x

    #DNN Forward
    def d_forward(self, x):
        x = self.test(x)
        return x

#CNN Model
net = Net().to(device)

#DNN CODE
#print(mlp(imge.to(device).view(28*28)))

#Run Train and Test with DNN
def run_epoch(model, train_data, test_data, optimizer, criterion):
    star_time = time.time()

    for img_i, label_i in train_data:

        img_i, label_i = img_i.to(device), label_i.to(device)

        optimizer.zero_grad()

        #Forward
        #label_predicted = mlp.forward(img_i.view(-1, 28 * 28))
        label_predicted = net.d_forward(img_i.view(-1, 28 * 28))
        #Loss computiation
        loss = criterion(label_predicted, label_i)
        #BackWard
        loss.backward()
        #Optimize for img_i
        optimizer.step()

    total_test_loss = 0
    for img_j, label_j in test_data:
        img_j, label_j = img_j.to(device), label_j.to(device)

        with torch.autograd.no_grad():
            #label_predicted = mlp.forward(img_j.view(-1, 28 * 28))
            label_predicted = net.forward(img_j.view(-1, 28 * 28))
            total_test_loss += criterion(torch.log(label_predicted), label_j)


    end_time = time.time()

    return total_test_loss, (end_time - star_time)

#Run Train and Test with CNN
def run_epochs(model, train_data, test_data, optimizer, criterion):
    star_time = time.time()

    trn_loss = 0.0
    for img_i, label_i in (train_data):

        img_i, label_i = img_i.to(device), label_i.to(device)


        optimizer.zero_grad()
        #Forward
        model_output = model.c_forward(img_i)
        print(model_output)

        #Loss
        loss = criterion(model_output, label_i)
        print(label_i)
        #BackWard
        loss.backward()
        #Optimizer for img_i
        optimizer.step()

    total_test_loss = 0
    for img_j, label_j in test_data:
        img_j, label_j = img_j.to(device), label_j.to(device)

        with torch.autograd.no_grad():
            # label_predicted = mlp.forward(img_j.view(-1, 28 * 28))
            label_predicted = net.c_forward(img_j)
            total_test_loss += criterion(torch.log(label_predicted), label_j)

    end_time = time.time()

    return total_test_loss, (end_time - star_time)


#Optimizer(최적화) and Criterion(손실함수)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

trn_loss_list = []
val_loss_list = []

for epoch in range(20):
    #code For un_batch image
    #test_loss, response = run_epoch(net, train_data, test_data, optimizer, criterion)
    #code For batch image
    #code For DNN
    #test_loss, response = run_epoch(net, train_data_laoder, test_data_laoder, optimizer, criterion)
    #code for CNN

    test_loss, response = run_epochs(net, train_data_laoder, test_data_laoder, optimizer, criterion)
    print('start epoch')
    if (epoch % 5 == 1):
        print('epoch ', epoch, ': ')
        print('\ttest_loss: ', test_loss)
        print('\tresponse(s): ', response)

vis_loader = torch.utils.data.DataLoader(test_data, 16, True)
img_vis, label_vis = next(iter(vis_loader))

imshow_grid(img_vis)

label_predicted = net.c_forward(img_vis.to(device).view(200, -1, 28 * 28))
print(label_predicted.shape)

_, top_i = torch.topk(label_predicted, k=1, dim=-1)

print('prediction: ', top_i.transpose(0, 1).cpu())
print('real label: ', label_vis.view(1, -1))
