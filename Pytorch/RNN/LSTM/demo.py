import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from model import Basic097, encoder
from torch import nn
from torch.utils.data import DataLoader

dataset = load_dataset("./stanford-imdb/plain_text")
tokenizer = AutoTokenizer.from_pretrained("./bert-tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 4
num_hiddens = max_length = 512

model = Basic097(tokenizer.vocab_size ,num_hiddens, device , max_length = max_length)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-3)

loss_epoch = []

data_loader = DataLoader(dataset["train"] ,batch_size = batch_size , shuffle=False)


def trainer(model , n_epochs):
    for epoch in range(n_epochs):
        loss_reg = 0
        state = model.begin_state(batch_size , device)
        for X,label in data_loader:
            optimizer.zero_grad()
            X = encoder(tokenizer,X,max_length=max_length)
            Y,state = model(X,state).to(device)
            l = loss(model.predict(Y.to(device)) , state.to(device))
            loss_reg += l
            l.backward()
            optimizer.step()

        loss_epoch.append(loss_reg.detech())
        print(f"epoch: {epoch} , loss: {loss_reg}")

trainer(model,2)




