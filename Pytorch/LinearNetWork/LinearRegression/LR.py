import torch
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils import data
from datasets import DatasetDict,Dataset,load_from_disk

data_path = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_from_disk("./data" )
dataset.set_format('torch')

batch_size = 16
x_loader = data.DataLoader(dataset["train"]["x_axis"] , batch_size=batch_size , shuffle=False)
y_loader = data.DataLoader(dataset["train"]["y_axis"].reshape(-1,1) , batch_size=batch_size , shuffle=False)

class LR(nn.Sequential):

    def __init__(self , in_dim , out_dim):
        super(LR , self).__init__()
        self.layer = nn.Linear(in_dim , out_dim)

    def forward(self,input):

        return self.layer(input)


module = LR(2,1)

def init_weights(layer):
    if layer == nn.Linear:
        nn.init.normal_(layer.weight , std=0.01)

module.apply(init_weights)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(module.parameters() , lr = 0.1)

loss_epoch = []

def trainer(module , n_epochs):

    for epoch in range(n_epochs):
        loss_reg = 0
        for x, y in zip(x_loader,y_loader):

            optimizer.zero_grad()
            output = module(x)
            l = loss(output , y)
            loss_reg +=l
            l.backward()
            optimizer.step()
        print(f"epoch:{epoch} , loss:{loss_reg}")
        loss_epoch.append(loss_reg.detach().numpy())

trainer(module , 5)

# print all params of module
'''
for param in module.parameters():
    print(param)
'''

def loss_show():

    plt.plot(range(len(loss_epoch)) , loss_epoch)
    plt.show()

loss_show()
