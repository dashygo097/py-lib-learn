import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pandas as pd

data_reg = pd.read_csv("./data/melon.csv")
print(data_reg)
data_reg = np.array(data_reg)


def sigmoid(x, theta):

    return 1 / (1 + np.exp(theta - x))


def d_sigmoid(x):

    return np.dot(x, 1 - x)


class Layer:

    def __init__(self):

        self.theta = None

    def init_params(self, q):

        self.theta = np.zeros((q))

    def forward(self, input):

        return sigmoid(input, self.theta)

    def backward(self, d_out, output, eta):

        g = d_sigmoid(output) * d_out
        self.theta -= eta * g

        return g


class BP_NetWork:

    def __init__(self):

        self.yLayer = Layer()
        self.hLayer = Layer()
        self.x_train = None
        self.y_train = None
        self.w = None
        self.v = None
        self.q = None

        self.n_sample = None
        self.eta = None

        self.loss = []

    def init_params(self, x_set, y_set, q, eta):

        self.x_train = np.array(x_set)
        self.y_train = np.array(y_set)
        if self.x_train.ndim > 1:
            xdim = self.x_train.shape[1]
        else:
            xdim = 1
        if self.y_train.ndim > 1:
            ydim = self.y_train.shape[1]
        else:
            ydim = 1
        self.w = np.zeros((q, ydim))
        self.v = np.zeros((xdim, q))
        self.n_sample = self.x_train.shape[0]
        self.eta = eta
        self.q = q
        self.yLayer.init_params(ydim)
        self.hLayer.init_params(q)

    def forward(self, x):

        hin = np.dot(x, self.v)
        x_n = x.shape[0]
        if x.ndim == 1:
            x_n = 1
        hout = []
        yout = []
        for i in range(x_n):
            hout.append(self.hLayer.forward(hin[i]))

        yin = np.dot(hout, self.w)
        for i in range(x_n):
            yout.append(self.yLayer.forward(yin[i]))

        yout = np.array(yout)
        hout = np.array(hout)

        return hout, yout

    def backward(self, batch_size):

        batch_index = random.randint(0, self.n_sample, size=(batch_size))
        x_sample = self.x_train[batch_index]
        y_sample = self.y_train[batch_index]
        hout, yout = self.forward(x_sample)

        for epoch in range(len(yout)):

            hout[epoch], yout[epoch] = self.forward(x_sample[epoch])
            d_out = y_sample[epoch] - yout[epoch]
            y = np.array(yout[epoch])
            b = np.array(hout[epoch])
            g = self.yLayer.backward(d_out, y, self.eta)
            e = self.hLayer.backward(d_out, b, self.eta)
            delta_w = self.eta * b[:, np.newaxis] * g[np.newaxis, :]
            delta_v = self.eta * x_sample[epoch][:, np.newaxis] * e[np.newaxis, :]
            self.w += delta_w
            self.v += delta_v
            self.loss.append(d_out)

    def get_params(self):

        return self.v, self.w

    def loss_show(self):

        ave_loss = []
        time = len(self.loss) // 100
        time_trained = range(time)
        for i in range(len(self.loss) // 100):
            ave_loss.append(np.mean(self.loss[i * 100 : (i + 1) * 100 :]))

        plt.plot(time_trained, ave_loss)
        plt.show()


x_train = np.array(data_reg[:, :-1:])
y_train = np.array(data_reg[:, -1])
invi_layer = 10

module = BP_NetWork()
module.init_params(x_train, y_train, invi_layer, 0.02)
module.backward(100000)
v, w = module.get_params()
hout, yout = module.forward(np.array([0.1, 0.8]))
print(yout)

module.loss_show()

exit()
