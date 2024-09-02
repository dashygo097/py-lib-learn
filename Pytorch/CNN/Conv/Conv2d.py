import torch
import os
import gzip
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from torch import nn

# cross-correlation operator
def corr2d(X,kernel):

    h,w = kernel.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * kernel).sum()
    return Y
'''
channel = torch.randint(0,2,size =(6,8))
kernel = torch.tensor([[1,2,3],[4,5,6]])


print(channel)
print(corr2d(channel , kernel))
'''
# Learning Conv Kernel

class Conv(nn.Sequential):

    def __init__(self ):
        super(Conv,self).__init__()
        self.conv2d = nn.Conv2d(1,1,kernel_size=(8,8),bias=False)

    def forward(self, input):

        return self.conv2d(input)

module = Conv()

loss = nn.MSELoss()
optimizer = torch.optim.SGD(module.parameters() , lr=0.1)

