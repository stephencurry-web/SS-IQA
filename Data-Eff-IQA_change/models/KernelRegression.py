import numpy as np

class KernelRegression:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = self._kernelize(distances / self.bandwidth)
            weighted_sum = np.dot(weights, self.y_train)
            y_pred.append(weighted_sum / weights.sum())
        return np.array(y_pred)

    def _kernelize(self, distances):
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * distances**2)
        elif self.kernel == 'linear':
            return 1 - distances
        elif self.kernel == 'quadratic':
            return (1 - distances)**2
        else:
            raise ValueError('Invalid kernel type')