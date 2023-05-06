import numpy as np

class LocallyWeightedRegression:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            weights = np.exp(-0.5 * ((self.X_train - x)**2).sum(axis=1) / self.bandwidth**2)
            weighted_sum = np.dot(weights, self.y_train)
            y_pred.append(weighted_sum / weights.sum())
        return np.array(y_pred)