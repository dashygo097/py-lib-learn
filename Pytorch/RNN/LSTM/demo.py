import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from model import Basic097, encoder
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

dataset = load_dataset("./stanford-imdb/plain_text")
tokenizer = AutoTokenizer.from_pretrained("./bert-tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
num_hiddens = max_length = 512

model = Basic097(tokenizer.vocab_size ,num_hiddens, device , max_length = max_length).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-3)

loss_epoch = []

features = encoder(tokenizer,dataset["train"]["text"],max_length=max_length)
labels = torch.tensor(dataset["train"]["label"])
train_data = TensorDataset(features, labels)
data_loader = DataLoader(train_data , batch_size=batch_size  ,shuffle=False)

def trainer(model, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        loss_reg = 0
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            state = model.begin_state(batch_size, device=device)
            output, state = model(X_batch.detach(), state)
            pred = model.predict(output, device)

            l = loss(pred, y_batch)
            loss_reg += l.item()

            l.backward()
            optimizer.step()

        loss_reg /= len(data_loader)
        print(f"epoch: {epoch}, loss: {loss_reg:.4f}")

trainer(model,5)




