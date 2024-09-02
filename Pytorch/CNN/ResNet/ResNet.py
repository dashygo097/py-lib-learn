import torch

import torch.nn.functional as F

from torch import nn



class Residual(nn.Module):

    def __init__(self , inputs_channels , output_channels,
            use_1x1conv=False , strides = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inputs_channels , output_channels ,
            kernel_size=3 , stride=strides , padding=1)
        self.BNLayer1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(inputs_channels , output_channels ,
            kernel_size=3 , stride=strides , padding=1)
        self.BNLayer2 = nn.BatchNorm2d(output_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(inputs_channels, output_channels,
                               kernel_size=3, stride=strides, padding=1)
        else:
            self.conv3 = None

    def forward(self , x):
        y = F.relu(self.conv1(self.BNLayer1(x)))
        y = self.conv2(self.BNLayer2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


