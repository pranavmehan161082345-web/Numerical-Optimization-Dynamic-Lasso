import numpy as np
class DynamicLassoOptimizer:
    def __init__(self, lr=0.01, lmbda=10, decay=0.01):
        self.lr, self.lmbda, self.decay, self.w = lr, lmbda, decay, None
    def _soft_threshold(self, x, thresh):
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
    def fit(self, X, y, iters=500):
        self.w = np.zeros(X.shape[1])
        for t in range(1, iters + 1):
            grad = (1/len(X)) * X.T @ (X @ self.w - y)
            curr_lmbda = self.lmbda * np.exp(-self.decay * t)
            self.w = self._soft_threshold(self.w - self.lr * grad, self.lr * curr_lmbda)
    def predict(self, X): return X @ self.w