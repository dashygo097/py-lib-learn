from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

from attention import MultiHeadedAttetion
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self ,d_model,dropout, max_length = 1024):
        super().__init__()
        self.max_length = max_length
        self.P = torch.zeros(size=(1 , max_length , d_model))
        x = torch.arange(max_length , dtype=torch.float32).reshape(-1,1) / torch.pow(
        10000,torch.arange(0, d_model , 2 , dtype=torch.float32) / d_model)
        self.P[:,:,::2] = torch.sin(x)
        self.P[:,:,1::2] = torch.cos(x)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):

        x = x + self.P[:,:x.shape[1],:].to(x.device)
        return self.dropout(x)

class FFN(nn.Module):

    def __init__(self,d_model ,d_inner, dropout,**kwargs):
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(d_model , d_inner)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(d_inner , d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.dropout(x)

class AddNorm(nn.Module):

    def __init__(self ,norm_shape,dropout,**kwargs):
       super().__init__(**kwargs)
       self.layernorm = nn.LayerNorm(norm_shape)
       self.dropout = nn.Dropout(dropout)

    def forward(self,x,y):
        return self.layernorm(self.dropout(y) + x)

class EncoderBlock(nn.Module):

    def __init__(self ,d_model,num_heads,d_inner,
        q_size,k_size,v_size,norm_shape,dropout,masked=None,bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = MultiHeadedAttetion(
            num_heads,d_model,q_size,k_size,v_size,dropout,masked=masked,bias=bias
            )
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = FFN(d_model,d_inner,dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self,x):
        y = self.addnorm1(x , self.attention(x,x,x))
        z = self.addnorm2(y , self.ffn(y))
        return z

X = torch.ones(size=(2, 100, 24))
encoder_blk = EncoderBlock(24, 8,48, 24,  24, 24,  [100,24],0.5)
encoder_blk.eval()
# input :(2,100,24)
# output:(2,100,24)
# In this case :
# d_model = 24 , Q=K=V=X with d_q=d_k=d_v=3

class Encoder(nn.Module):

    def __init__(self,vocab_size,num_layers,d_model,num_heads,d_inner,q_size,
        k_size,v_size,norm_shape,dropout,max_length=1024,masked=None,bias=False,**kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = PositionalEncoding(d_model,dropout,max_length=max_length)
        self.embed = nn.Embedding(vocab_size,d_model)
        self.d_model = d_model
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("Encoder"+str(i),EncoderBlock(
                d_model,num_heads,d_inner,q_size,k_size,v_size,
                norm_shape,dropout,masked=masked,bias=bias))

    def forward(self):
        
        return




