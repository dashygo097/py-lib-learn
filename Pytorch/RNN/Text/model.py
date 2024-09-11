import torch

import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from txtloader import load_txt

data,vocab = load_txt("./data/timemachine.txt",flatten=True,token="char")

class RNNModel(nn.Module):

    def __init__(self):
        super().__init__()


x = torch.tensor([1,2], dtype=torch.float32)
print(F.softmax(1*x, dim=0))
print(F.softmax(2*x, dim=0))