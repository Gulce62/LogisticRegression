import numpy as np
import copy
from sklearn.metrics import accuracy_score


def gaussianInitialization(dimension):
    np.random.seed(9)
    w = np.random.normal(loc=0, scale=1, size=(dimension, 1))
    return w


def zeroInitialization(dimension):
    np.random.seed(9)
    w = np.zeros((dimension, 1))
    return w


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


class LogisticRegression:
    def __init__(self, epochs=100, learningRate=0.001, batchSize=64):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize

    def fit(self, X_train, y_train):
        m = X_train.shape[0]
        w = zeroInitialization(X_train.shape[1])
        b = zeroInitialization(1)
        for epoch in range(self.epochs):
            for batch in range(m // self.batchSize + 1):
                startIdx = batch * self.batchSize
                endIdx = 2 * batch * self.batchSize
                if batch == m // self.batchSize:
                    prob = sigmoid(np.dot(X_train[startIdx:endIdx], w) + b)
                    dw = (1 / self.batchSize) * np.dot(X_train[startIdx:endIdx].T, (prob - y_train[startIdx:endIdx]))
                    db = (1 / self.batchSize) * np.sum(prob - y_train[startIdx:endIdx])
                else:
                    prob = sigmoid(np.dot(X_train[startIdx:], w) + b)
                    dw = (1 / self.batchSize) * np.dot(X_train[startIdx:].T, (prob - y_train[startIdx:]))
                    db = (1 / self.batchSize) * np.sum(prob - y_train[batch])
                w -= self.learningRate * dw
                b -= self.learningRate * db
            print('-------------- Epoch', epoch, 'finished --------------')
        return w, b

    def predict(self, w, b, X, threshold):
        y_pred = np.zeros(X.shape[0])
        prob = sigmoid(np.dot(X, w) + b)
        for i in range(prob.shape[0]):
            if prob[i, 0] > threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
