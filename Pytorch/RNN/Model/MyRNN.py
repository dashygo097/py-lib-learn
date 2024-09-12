import torch

import torch.nn.functional as F

from torch import nn

def rnn_cal(inputs,state,params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs= []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

class Basic097(nn.Module):

    def __init__(self, vocab_size , num_hiddens, out_size,device,
    forward_fn=rnn_cal , init_state=init_rnn_state):
        super().__init__()
        self.inputs_size = vocab_size
        self.outputs_size = out_size
        self.num_hiddens = num_hiddens

        self.W_xh = (torch.normal(0, 0.01, size=(vocab_size, num_hiddens),device=device)).requires_grad_(True)
        self.W_hh = (torch.normal(0, 0.01, size=(num_hiddens, num_hiddens),device=device)).requires_grad_(True)
        self.W_hq = (torch.normal(0, 0.01, size=(num_hiddens, out_size),device=device)).requires_grad_(True)
        self.b_h = (torch.zeros(size=(num_hiddens,), device=device)).requires_grad_(True)
        self.b_q = (torch.zeros(size=(out_size,), device=device)).requires_grad_(True)

        self.forward_fn = forward_fn
        self.init_state = init_state
        self.encoder = F.one_hot

        self.params = [self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q]


    def forward(self,inputs,state):
        embeded_inputs = self.encoder(inputs.transpose(1,0),self.inputs_size).type(torch.float32)

        return self.forward_fn(embeded_inputs , state , self.params)

    def begin_state(self, batch_size , device):
        return self.init_state(batch_size, self.num_hiddens,device)



