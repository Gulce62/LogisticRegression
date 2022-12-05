import numpy as np


def sigmoid(z):
    # Sigmoid function
    s = 1 / (1 + np.exp(-z))
    return s


def gaussianInitialization(dimension):
    # Initialize weights with gaussian distribution
    np.random.seed(50)
    p = np.random.normal(loc=0, scale=1, size=(dimension, 1))
    return p


def uniformInitialization(dimension):
    # Initialize weights with uniform distribution
    np.random.seed(50)
    p = np.random.uniform(size=(dimension, 1))
    return p


def zeroInitialization(dimension):
    # Initialize weights with zero
    p = np.zeros((dimension, 1))
    return p


def getAccuracy(y_true, y_pred):
    # Get the accuracy
    accuracyBool = (y_true.ravel() == y_pred.ravel())
    accuracy = np.count_nonzero(accuracyBool) / accuracyBool.shape[0]
    return accuracy


class LogisticRegression:
    def __init__(self, epochs=100, learningRate=0.001, batchSize=64, initialization='gaussian'):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.initialization = {'gaussian': lambda dimension: gaussianInitialization(dimension),
                               'uniform': lambda dimension: uniformInitialization(dimension),
                               'zero': lambda dimension: zeroInitialization(dimension)}[initialization]

    def fit(self, X_train, y_train, X_val, y_val):
        # Train the logistic function
        m = X_train.shape[0]
        # Initialize weights
        w = self.initialization(X_train.shape[1])
        b = self.initialization(1)
        # Find the iteration number according to batch size
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1
        val_acc = []
        for epoch in range(self.epochs):  # for every epoch
            for batch in range(iterationNo):  # for every batch
                # Calculate batch indices
                startIdx = batch * self.batchSize
                endIdx = startIdx + self.batchSize
                # The final batch (it cannot be the same size as others)
                if batch == iterationNo - 1:
                    prob = sigmoid(np.dot(X_train[startIdx:], w) + b)
                    dw = np.dot(X_train[startIdx:].T, (prob - y_train[startIdx:]))
                    db = np.sum(prob - y_train[startIdx:])
                # All other batches
                else:
                    prob = sigmoid(np.dot(X_train[startIdx:endIdx], w) + b)
                    dw = np.dot(X_train[startIdx:endIdx].T, (prob - y_train[startIdx:endIdx]))
                    db = np.sum(prob - y_train[startIdx:endIdx])
                # Update rule
                w -= self.learningRate * dw
                b -= self.learningRate * db
            # Calculate validation accuracy for every epoch
            y_pred = self.predict(w, b, X_val, 0.5)
            val_accuracy = getAccuracy(y_pred, y_val)
            val_acc.append(val_accuracy)
            print('Epoch', epoch, 'finished. Accuracy is', val_accuracy)
        return w, b, val_acc

    def predict(self, w, b, X, threshold):
        # Predict the examples with logistic function and founded weights
        y_pred = np.zeros(X.shape[0])
        prob = sigmoid(np.dot(X, w) + b)
        for i in range(prob.shape[0]):
            if prob[i, 0] > threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
