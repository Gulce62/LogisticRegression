import numpy as np
import copy

def gaussianInitialization():
    pass

def zeroInitialization(dimension):
    w = np.zeros((dimension, 1))
    return w

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

class LogisticRegression:
    def __init__(self, epochs, learningRate):
        self.epochs = epochs
        self.learningRate = learningRate

    def fit(self, X_train, y_train):
        w = zeroInitialization(X_train.shape[1])
        m = X_train.shape[0]
        print(w.shape)
        for e in range(self.epochs):
            prob = 1-sigmoid(np.dot(X_train, w))
            dw = (1 / m) * np.dot(X_train.T, (y_train-prob))
            w += self.learningRate * dw
        return w

    def predict(self, w, X):
        m = X.shape[0]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[1], 1)
        A = sigmoid(np.dot(X, w))
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        return Y_prediction
