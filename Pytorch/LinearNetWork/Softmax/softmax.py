import torch
import gzip
import numpy as np
import numpy.random as random
import os
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

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

    fig, ax = plt.subplots(ncols=8, nrows=2, figsize=(28, 28))
    index = random.randint(60000, size=(16))
    img = mnist_dataset["training_images"][index].reshape(-1, 28, 28)
    label = mnist_dataset["training_labels"][index]
    for i in range(16):
        ax[int(i / 8), i % 8].imshow(img[i], cmap="gray")
        ax[int(i / 8), i % 8].set_title(label[i])

    plt.show()


def fetch_train_data(x_train, y_train, batch_size):

    index = random.randint(60000, size=(batch_size))
    img = x_train[index]
    label = y_train[index]

    return torch.tensor(img), torch.tensor(label)


net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

mnist_dataset["training_images"] = torch.tensor(
    mnist_dataset["training_images"], dtype=torch.float
)
mnist_dataset["training_labels"] = torch.tensor(
    mnist_dataset["training_labels"], dtype=torch.float
)
x_train_loader = DataLoader(
    mnist_dataset["training_images"], batch_size=16, shuffle=False
)
y_train_loader = DataLoader(
    mnist_dataset["training_labels"], batch_size=16, shuffle=False
)


def train(net, num_epochs):

    for epoch in range(num_epochs):
        for img, label in zip(x_train_loader, y_train_loader):

            label = label.long()
            optimizer.zero_grad()
            output = net(img)
            l = loss(output, label)
            l.backward()
            optimizer.step()


train(net, 2)

mnist_dataset["test_images"] = torch.tensor(
    mnist_dataset["test_images"], dtype=torch.float
)
mnist_dataset["test_labels"] = torch.tensor(
    mnist_dataset["test_labels"], dtype=torch.float
)
x_train_loader = DataLoader(mnist_dataset["test_labels"], batch_size=16, shuffle=True)
y_train_loader = DataLoader(mnist_dataset["test_labels"], batch_size=16, shuffle=True)


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


img_show()
