import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        self.X_tr = None
        self.y_tr = None

    def train(self, X, y):
        """
        Nearest neighbor keeps track of all training examples and associated labels
        :param X: n by d numpy array (n examples of size d)
        :param y: 1 by d numpy array
        """
        self.X_tr = X
        self.y_tr = y

    def predict(self, X):
        if self.y_tr is not None and self.y_tr.shape[0] > 0:
            nb_examples = X.shape[0]
            y_pred = np.zeros(nb_examples, dtype=self.y_tr.dtype)
            for i in range(nb_examples):
                X_pred = X[i, :]
                distances = np.sum(np.abs(self.X_tr - X_pred), axis=1)
                min_idx = np.argmin(distances)
                y_pred[i] = self.y_tr[min_idx]
            return y_pred
        else:
            raise Exception('No neighbor provided')
