import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as random

data_reg = pd.read_csv("./data/melon.csv")
print(data_reg)


def ReLU(x):

    if x < 0:
        return 0
    else:
        return np.array(x)


def d_ReLU(x):

    x = x > 0
    return np.where(x, 1, 0)


class SVM:

    def __init__(self, tol=1e-3):

        self.w = None
        self.b = None
        self.alpha = None
        self.loss = None
        self.lam = None
        self.eta = None

        self.kernel = None
        self.tol = tol

        self.x_train = None
        self.y_train = None

        self.n_sample = None
        self.n_feature = None
        self.n_label = None
        self.loss_epoch = []
        self.acc_epoch = []

    def init_params(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        self.lam = 0.005
        self.eta = 0.1

        self.n_sample, self.n_feature = x_train.shape
        if y_train.ndim == 1:
            self.n_label = 1
        else:
            self.n_label = y_train[1]

        self.w = random.normal(0, 0.01, size=(self.n_feature))
        self.b = np.array(0.0)

        self.alpha = np.zeros(self.n_feature)
        self.loss = 1

    def forward(self, x):

        if x.ndim > 1:
            num = x.shape[0]
        else:
            num = 1

        return np.dot(x, self.w) + self.b

    def get_loss(self, x_data, y_data):

        loss = 0

        if x_data.ndim > 1:
            for x, y in zip(x_data, y_data):

                fx = self.forward(x)
                loss += ReLU(1 - y * fx) / x_data.shape[0]
        else:
            fx = self.forward(x_data)
            loss = ReLU(1 - y_data * fx)

        self.loss = loss + self.lam * np.dot(self.w, self.w)

        return self.loss

    def backward(self, eta, batch_size):

        batch_index = random.randint(self.n_sample, size=batch_size)
        x_batch = x_train[batch_index]
        y_batch = y_train[batch_index]

        loss = self.get_loss(x_batch, y_batch)
        self.loss_epoch.append(loss)
        self.eta = eta

        d_Lw = np.zeros(self.n_feature)
        d_Lb = 0

        for i in range(batch_size):
            d_Lw += (
                -y_batch[i]
                * x_batch[i]
                * d_ReLU(1 - y_batch[i] * self.forward(x_batch[i]))
                + 2 * self.lam * self.w
            ) / batch_size
        d_Lb = (
            np.sum(-y_batch * d_ReLU((1 - y_batch * self.forward(x_batch))))
        ) / batch_size

        self.w -= self.eta * d_Lw
        self.b -= self.eta * d_Lb

    def train(self, x_train, y_train, n_epochs, batch_size, eta):

        self.init_params(x_train, y_train)
        self.eta = eta

        for i in range(n_epochs):

            self.backward(eta, batch_size)

    def get_params(self):

        return self.w, self.b

    def show_loss(self):

        plt.plot(range(len(self.loss_epoch)), self.loss_epoch)
        plt.show()

    def decide(self, x_test, y_test):

        y_get = np.sign(self.forward(x_test))
        print(self.forward(x_test[:20:]))
        print(y_get[:20:])
        equal = np.equal(y_get, y_test)
        print(y_test[:20:])

        acc = np.sum(np.where(equal, 1, 0)) / x_test.shape[0]

        return acc


data_reg = np.array(data_reg)
x_train = data_reg[:, :-1:]
y_train = data_reg[:, -1]


module = SVM()

module.train(x_train, y_train, 3000, 32, 0.06)

w, b = module.get_params()
print(w)
print(b)

print(module.decide(x_train[:1000:, :], y_train[:1000:]))

module.show_loss()
