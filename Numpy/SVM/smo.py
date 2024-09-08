import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as random

data_reg = pd.read_csv("./data/melon.csv")
print(data_reg)


def sigmoid(x):

    return np.array(
        (np.exp(10 * x) - np.exp(-10 * x)) / (np.exp(10 * x) + np.exp(-10 * x))
    )


def ReLU(x):

    if x < 0:
        return 0
    else:
        return np.array(x)


def d_ReLU(x):

    x = x > 0
    return np.where(x, 1, 0)


def linear_kernel(x1, x2):

    return np.dot(x1, x2)


def gauss_kernel(x1, x2, sigma):
    ga = np.sum((x1 - x2) ** 2)
    ga = np.exp(-ga / (2 * sigma**2))
    return ga


class SVM:

    def __init__(self, tol=1e-3, C=2):

        self.w = None
        self.b = None
        self.alpha = None
        self.loss = None
        self.lam = None
        self.eta = None
        self.C = C
        self.E = None
        self.g = None
        self.is_kkt = None

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
        self.eta = 0.05

        self.n_sample, self.n_feature = x_train.shape
        if y_train.ndim == 1:
            self.n_label = 1
        else:
            self.n_label = y_train[1]

        self.w = random.normal(0, 0.01, size=(self.n_feature))
        self.b = random.normal(0, 0.01)

        self.alpha = np.zeros(self.n_sample)
        self.E = random.normal(0, 0.01, size=self.n_sample)
        self.g = np.zeros(self.n_sample)
        self.is_kkt = np.zeros(self.n_sample)
        self.kkt_count = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            self.is_kkt[i] = False
        self.loss = 1

    def fit(self):

        self.w = self.alpha * self.y_train * x_train
        tobe = np.sum(self.alpha * self.y_train)

        return tobe == 0

    def really_fit(self):

        flag = self.fit()
        if flag:
            out_set = self.forward(x_train)
            for out, y_i, alpha_i in zip(out_set, self.y_train, self.alpha):
                flag = flag & ((out * y_i >= 1 - self.tol) | (alpha_i == 0))

        return flag

    def forward(self, x):

        if x.ndim > 1:
            num = x.shape[0]
        else:
            num = 1

        return np.dot(x, self.w) + self.b

    def get_loss(self):

        self.g = np.zeros(self.n_sample)

        for i in range(self.n_sample):
            for j in range(self.n_sample):
                self.g[i] += (
                    self.alpha[j]
                    * self.y_train[j]
                    * gauss_kernel(self.x_train[j], self.x_train[i], 2)
                )
            self.g[i] += self.b
            self.E[i] = self.g[i] - self.y_train[i]

    def clip(self, i, j, alpha_j_dnf):

        if self.y_train[i] != self.y_train[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])

        if alpha_j_dnf > H:
            return H
        elif alpha_j_dnf < L:
            return L
        else:
            return alpha_j_dnf

    def smo(self, i, j):

        self.E = self.forward(x_train) - y_train
        x_i = self.x_train[i]
        x_j = self.x_train[j]
        y_i = self.y_train[i]
        y_j = self.y_train[j]
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]

        k_ii = gauss_kernel(x_i, x_i, 0.001)
        k_ij = gauss_kernel(x_i, x_j, 0.001)
        k_jj = gauss_kernel(x_j, x_j, 0.001)

        eta = k_ii + k_jj - 2 * k_ij
        if eta == 0:
            eta = self.tol
        alpha_j_dnf = alpha_j_old + y_j * (self.E[i] - self.E[j]) / eta
        alpha_j_new = self.clip(i, j, alpha_j_dnf)
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.E[i] = self.forward(x_i) - y_i
        self.E[j] = self.forward(x_j) - y_j

        b_i_new = (
            -self.E[i]
            - y_i * k_ii * (alpha_i_new - alpha_i_old)
            - y_j * k_ij * (alpha_j_new - alpha_j_old)
            + self.b
        )
        b_j_new = (
            -self.E[j]
            - y_i * k_ij * (alpha_i_new - alpha_i_old)
            - y_j * k_jj * (alpha_j_new - alpha_j_old)
            + self.b
        )
        if 0 < self.alpha[i] < self.C:
            self.b = b_i_new
        elif 0 < self.alpha[j] < self.C:
            self.b = b_j_new
        else:
            self.b = (b_i_new + b_j_new) / 2.0

        self.w = np.dot(self.alpha * self.y_train, self.x_train)

    def get_kkt(self, i, j):

        if self.alpha[i] == 0:
            self.is_kkt[i] = self.y_train[i] * self.g[i] >= 1
        elif self.alpha[i] == self.C:
            self.is_kkt[i] = self.y_train[i] * self.g[i] <= 1
        else:
            self.is_kkt[i] = self.y_train[i] * self.g[i] == 1

        if self.alpha[j] == 0:
            self.is_kkt[j] = self.y_train[j] * self.g[j] >= 1
        elif self.alpha[j] == self.C:
            self.is_kkt[j] = self.y_train[j] * self.g[j] <= 1
        else:
            self.is_kkt[j] = self.y_train[j] * self.g[j] == 1

        if self.is_kkt[i] == False:
            self.kkt_count[i] += 1
        if self.kkt_count[i] > 5:
            self.is_kkt[i] = True

    def choose_i(self):

        for i in range(self.n_sample):
            if self.is_kkt[i] == False:
                return i
        return -1

    def choose_j(self, i):

        if self.E[i] >= 0:
            j = np.argmin(self.E)
        if self.E[i] < 0:
            j = np.argmax(self.E)
        return j

    def train(self, x_train, y_train, n_epochs):

        self.init_params(x_train, y_train)

        for epoch in range(n_epochs):

            i = self.choose_i()
            j = self.choose_j(i)
            self.smo(i, j)
            self.get_kkt(i, j)

    def decide(self, x_test, y_test):

        y_out = np.sign(self.forward(x_test))

        print(self.forward(x_test[:10:]))
        print(y_out[:10:])
        equal = np.equal(y_out, y_test)
        print(y_test[:10:])

        acc = np.sum(np.where(equal, 1, 0)) / x_test.shape[0]

        return acc


data_reg = np.array(data_reg)
x_train = data_reg[:, :-1:]
y_train = data_reg[:, -1]


module = SVM()

module.train(x_train, y_train, 30000)
acc = module.decide(x_train[:100:], y_train[:100:])
print(acc)
