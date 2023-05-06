from collections import Counter
import numpy as np

class KneighborsRegressor:
    # def __init__(self, k):
    #     self.k = k
    #
    # def fit(self, X, y):
    #     self.X_train = X
    #     self.y_train = y
    #
    # def predict(self, X):
    #     predictions = np.zeros(len(X), dtype=self.y_train.dtype)
    #     for i, x in enumerate(X):
    #         distances = []
    #         for j in range(len(self.X_train)):
    #             distance = np.sqrt(np.sum(np.square(x - self.X_train[j])))
    #             distances.append((distance, self.y_train[j]))
    #         distances = sorted(distances)[:self.k]
    #         labels = [d[1] for d in distances]
    #         most_common_label = max(set(labels), key=labels.count)
    #         predictions[i] = most_common_label
    #     return predictions
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = np.sqrt(((self.X_train - X) ** 2).sum(axis=1))
        indices = distances.argsort()[:self.k]
        return np.mean(self.y_train[indices])