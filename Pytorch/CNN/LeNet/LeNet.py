import torch
import gzip
import numpy as np
import numpy.random as random
import os
import matplotlib.pyplot as plt

from BatchNorm import BatchNorm
from torch import nn
from torch.utils.data import DataLoader

#gpu

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]



data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}


mnist_dataset = {}

# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)


def img_show():

    # x_train_iter = data.DataLoader(mnist_dataset["training_images"], batch_size=16)
    # y_train_iter = data.DataLoader(mnist_dataset["training_labels"], batch_size=16)

    fig, ax = plt.subplots(ncols=8, nrows=2, figsize=(8, 5.6))
    index = random.randint(60000, size=(16))
    img = mnist_dataset["training_images"][index].reshape(-1, 28, 28)
    label = mnist_dataset["training_labels"][index]
    for i in range(16):
        ax[int(i / 8), i % 8].imshow(img[i], cmap="gray")
        ax[int(i / 8), i % 8].set_title(f'Label:{label[i]}')

    plt.show()



net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),BatchNorm(6,num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),BatchNorm(16,num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),BatchNorm(120,num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84,num_dims=2),nn.Sigmoid(),
    nn.Linear(84, 10))

acc_epoch = []
loss_epoch = []
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=1)


net.apply(init_weights)
net.to(device='cuda:0')

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

mnist_dataset["training_images"] = torch.tensor(
    mnist_dataset["training_images"].reshape(-1,1,28,28), dtype=torch.float
).cuda()
mnist_dataset["training_labels"] = torch.tensor(
    mnist_dataset["training_labels"], dtype=torch.float
).cuda()
mnist_dataset["test_images"] = torch.tensor(
    mnist_dataset["test_images"].reshape(-1,1,28,28), dtype=torch.float
).cuda()
mnist_dataset["test_labels"] = torch.tensor(
    mnist_dataset["test_labels"], dtype=torch.float
).cuda()
x_train_loader = DataLoader(
    mnist_dataset["training_images"], batch_size=256, shuffle=False
)
y_train_loader = DataLoader(
    mnist_dataset["training_labels"], batch_size=256, shuffle=False
)

def train(net, num_epochs):

    for epoch in range(num_epochs):
        right_judge = 0
        loss_sum = 0
        count = 0
        for img, label in zip(x_train_loader, y_train_loader):

            label = label.long()
            optimizer.zero_grad()
            output = net(img)
            l = loss(output, label)
            loss_sum += l
            count+= 1
            l.backward()
            optimizer.step()

        loss_epoch.append(loss_sum.to('cpu').detach().numpy() / count )
        labels = mnist_dataset["test_labels"]
        result, result_labels = torch.max(
            net.forward(mnist_dataset["test_images"].cuda()), 1
        )
        right_judge = labels == result_labels.int()
        right_judge = torch.sum(right_judge, axis=0)
        acc = right_judge.cpu().detach().numpy() / mnist_dataset["test_images"].shape[0]
        acc_epoch.append(acc)


n_epochs = 12
train(net, n_epochs)

net.eval()

def img_show():

    # x_train_iter = data.DataLoader(mnist_dataset["training_images"], batch_size=16)
    # y_train_iter = data.DataLoader(mnist_dataset["training_labels"], batch_size=16)

    fig, ax = plt.subplots(ncols=8, nrows=2, figsize=(10, 10))
    index = random.randint(60000, size=(16))
    img = mnist_dataset["training_images"][index].reshape(-1, 28, 28)
    label = mnist_dataset["training_labels"][index]
    result, result_label = torch.max(
        net.forward(mnist_dataset["training_images"][index]), 1
    )
    for i in range(16):
        ax[int(i / 8), i % 8].imshow(img[i], cmap="gray")
        label_show = str(int(label[i]))
        result_show = str(int(result_label[i]))
        ax[int(i / 8), i % 8].set_title("".join([label_show, "\n", result_show]))

    plt.show()


def show_eval():
    plt.plot(range(len(loss_epoch)), loss_epoch, color='#48A43F' , label='loss')
    plt.plot(range(len(acc_epoch)), acc_epoch, color='orange' ,label='accuracy')

    plt.xlabel('epoch')
    plt.title("Loss & Accuracy Trend")
    plt.legend()
    plt.grid()
    plt.show()

print(f'loss:{loss_epoch}')
print(f'acc:{acc_epoch}')
show_eval()


