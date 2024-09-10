import torch

import numpy.random as random
import torch.nn.functional as F

from torch import nn


class Word2Vec(nn.Module):

    def __init__(self , vocab_size , embed_size,get_input_same = True):
        super().__init__()
        self.get_input_same = get_input_same
        if get_input_same :
            self.embed = nn.Embedding(vocab_size , embed_size)
        else :
            self.embed_u = nn.Embedding(vocab_size , embed_size)
            self.embed_v = nn.Embedding(vocab_size , embed_size)

    def forward(self,center , contexts):

        if self.get_input_same == False:
            u = self.embed_u(center)
            v = self.embed_v(contexts)
            pred = torch.bmm(v,u.transpose(-1,-2))

        else :
            u = self.embed(center)
            v = self.embed(contexts)
            pred = torch.bmm(v,u.transpose(-1,-2))

        return pred

class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input , target , mask =None ):

        loss =0. - F.binary_cross_entropy_with_logits(input , target,weight=mask)

        return loss.mean(dim=-1)
