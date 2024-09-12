import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from torch import nn

batch_size = 4
seq_length = 2048
noise = torch.normal(0.,0.04,size=(seq_length,))
x_t = torch.sin(torch.arange(0,seq_length,1,dtype=torch.float32) / 100) + noise
features = torch.tensor([x_t[i:i + batch_size:].tolist()
    for i in range(x_t.shape[0] - batch_size)],dtype=torch.float32)
labels = x_t[batch_size::]

def show_pattern(series,label):

    plt.plot(range(series.shape[0]) ,series , label=label)
    plt.xlabel("time")
    plt.title("Time Domain Pattern")

class PredictPattern(nn.Module):

    def __init__(self , batch_size):
        super().__init__()
        self.layer1 = nn.Linear(batch_size,10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10,1)

    def forward(self,x):

        return self.layer2(self.relu(self.layer1(x)))

def init_weights(layer):
    if layer == nn.Linear:
        nn.init.normal_(layer.weight,std=0.01)

x_loader = data.DataLoader(dataset=features,batch_size=batch_size)
y_loader = data.DataLoader(dataset=labels,batch_size=batch_size)
model = PredictPattern(batch_size)
model.apply(init_weights)
loss_epoch = []
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr= 0.1)

def trainer(model , data_loader,n_epochs):
    for epoch in range(n_epochs):
        loss_tmp = 0
        for batched in data_loader:
            optimizer.zero_grad()
            output = model(batched)
            l = loss(output,batched)
            loss_tmp += l / batch_size
            l.backward()
            optimizer.step()
        loss_epoch.append(loss_tmp.detach().numpy())

trainer(model,x_loader , 10)

model.eval()
def show_loss():

    plt.plot(range(len(loss_epoch)),loss_epoch,label="loss")
    plt.title("Loss")
    plt.xlabel("epoch")

def img_show():
    show_pattern(labels, "original")
    show_pattern((model(features).detach().numpy()),"predicted")
    plt.legend()
    plt.show()
    show_loss()
    plt.legend()
    plt.show()

img_show()