import torch
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from datasets import load_from_disk
from torch import nn
from torch.utils import data
from word2vec import SigmoidBCELoss,Word2Vec #,get_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_from_disk("./data")
dataset.set_format("torch")

train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

batch_size = 16

train_loader = data.DataLoader(train_data , batch_size, shuffle=False)

vocab_dir = {}
index = 0

for sentence in train_data["sentence"]:
    word_dir = sentence.split(" ")
    for word in word_dir :
        if word not in vocab_dir.keys():
            vocab_dir[word] = index
            index += 1



def vocab_to_id(sentences):
    id_output = []
    for sentence in sentences:
        word_list = sentence.split(" ")
        id_list = []
        for word in word_list:
            id_list.append(vocab_dir[word])
        id_output.append(id_list)
    return id_output

embed_size = 100

model = Word2Vec(len(vocab_dir) , embed_size)

loss = SigmoidBCELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=1e-4)

loss_epoch = []

def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

'''
def trainer(model , n_epochs):

    model.to(device)
    for epoch in range(n_epochs):
        loss_accu = 0
        for sentence in train_loader:
            optimizer.zero_grad()
            word_id =vocab_to_id(sentence["sentence"])
            centers,contexts,labels = get_samples(word_id , 64)
            pred = model(centers,contexts)
            l = loss(pred.float() , labels.float())
            loss_accu = l.sum().detach().numpy()
            l.sum().backward()
            optimizer.step()

        loss_epoch.append(loss_accu)

trainer(model,5)
'''
def loss_show():
    plt.plot(range(len(loss_epoch)) , loss_epoch)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Trend of My Embedding Layer")
    plt.show()

loss_show()








