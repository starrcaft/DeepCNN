from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np

import matplotlib.pyplot as plt

#gpu-server setting
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#torch 기초 공부

#기초적인 배열 선언
v= torch.Tensor([1, 2, 3])
print (v)

#타입 확인
#타입은 int float 등등 기존에 알던 기본 타입을 의미
print(v.dtype)

#tensor는 파이선 배열과 마찬가지로 슬라이싱 가능
v = torch.tensor([1, 2, 3, 4, 5, 6])
print(v[1:])
print(v[1:4])

#float형태로 선언
f = torch.FloatTensor([1, 2, 3, 4, 5, 6])
print(f)

#크기 확인
print(f.size())

## 중요
#view 를 써서 배열 형태를 조작 가능
#자주 쓰이는 함수이며 이를 이용해 보기 쉽게 배열을 정리
print(v.view(6, 1))

# -1 을 통해 인자에 3만 신경쓰고 나머지는 인풋에 맞게 조절
print(v.view(3, -1))

#numpy array를 tensor array를 상호간에 변환하는 것이 가능
#이를 통해 numpy 인풋을 torch로 쉽게 변형 가능
a = np.array([1, 2, 3, 4, 5, 6])
tensor_cnv = torch.from_numpy(a)
print(tensor_cnv, tensor_cnv.type())

numpy_cnv = tensor_cnv.numpy()
print(numpy_cnv)

#0 부터 10 을 100개로 쪼갠다
print(torch.linspace(0, 10))
#0 부터 10을 5개로 쪼갠다.
print(torch.linspace(0, 10, 5))

#n차원 배열
one_d = torch.arange(0, 9)
two_d = one_d.view(3, 3)
print (two_d)

#dim으로 차원 확인 가능
print(two_d.dim())

#2개의 블록, 3로우, 3컬럼의 형태로 만듬
x = torch.arange(18).view(2, 3, 3)
print(x)
