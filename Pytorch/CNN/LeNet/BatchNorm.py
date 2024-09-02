import torch
from torch import nn


def batch_norm(X , gamma ,beta , moving_mean , moving_var ,epsilon ,momentum):
    # train mode or eval mode
    if torch.is_grad_enabled():
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + epsilon)

    else :
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2 :
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else :
            mean = X.mean(dim=(0 ,2 ,3), keepdim= True)
            var = ((X - mean) ** 2).mean(dim=(0 ,2 ,3), keepdim= True)

        X_hat = (X - mean) / torch.sqrt(var + epsilon)
        moving_mean = momentum * moving_mean + (1. - momentum) * mean
        moving_var = momentum * moving_var + (1. - momentum) * var

    Y = gamma * X_hat + beta
    return Y,moving_mean.data,moving_var.data

class BatchNorm(nn.Module):

    def __init__(self, num_features , num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1 , num_features)
        else :
            shape = (1, num_features , 1 , 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self , x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var  = self.moving_var.to(x.device)

        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma , self.beta , self.moving_mean ,
            self.moving_var , epsilon=1e-5 ,momentum=0.9
            )
        return y
