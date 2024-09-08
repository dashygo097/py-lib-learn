import gzip
import numpy as np
import numpy.random as random
import os
import matplotlib.pyplot as plt

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
