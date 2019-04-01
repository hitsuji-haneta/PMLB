import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                self.cost_.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    # def net_input(self, X):
    #     return np.dot(X, self.w_[1:]) + self.w_[0]

    # def activation(self, X):
    #     return X

    # def predict(self, X):
    #     return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
