import numpy as np

class BinaryLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

class LogisticRegressionScratchOvR:
    def __init__(self, lr=0.1, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.models = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            model = BinaryLogisticRegression(lr=self.lr, num_iter=self.num_iter)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        probs = np.column_stack([model.predict_proba(X) for model in self.models])
        return np.argmax(probs, axis=1)