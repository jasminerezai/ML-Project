import numpy as np

class BinaryLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr       #Learning rate
        self.num_iter = num_iter    #Number of iterations over dataset
        self.weights = None 
        self.bias = None

    def _sigmoid(self, z):
        # Maps raw score z to range (0, 1). np.clip avoids math overflow errors.
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        # Get the number of sampels(rows) and features (columns) in the dataset. 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.num_iter):
            #Calculate the linear combination of inputs, weights and bias.
            linear_model = np.dot(X, self.weights) + self.bias

            # Probability prediction via sigmoid activation
            y_predicted = self._sigmoid(linear_model)

            # Calculate the gradient for the weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias using gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

class LogisticRegressionScratchOvR:
    def __init__(self, lr=0.1, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.models = []   # Stores 3 trained binary models
        self.classes = []  # Unique class labels [0, 1, 2]

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        # Train 1 binary classifier for each species
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)  # Current class = 1, all others = 0
            model = BinaryLogisticRegression(lr=self.lr, num_iter=self.num_iter)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        # Stack probabilities from all 3 sub-models horizontally --> Shape: (N, 3)
        probs = np.column_stack([model.predict_proba(X) for model in self.models])
        # Pick class index with maximum probability score
        return np.argmax(probs, axis=1)