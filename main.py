import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score
import LogisticRegression as LR


def read_data(file_name):
    # Read the data with its path location
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception:
        sys.exit(1)


def trainValTestSplit(data):
    shuffled = data.sample(frac=1, random_state=0)
    dataSize = len(shuffled)
    train = shuffled[:int(dataSize * 0.7)]
    val = shuffled[int(dataSize * 0.7):int(dataSize * 0.8)]
    test = shuffled[int(dataSize * 0.8):]
    return train, val, test

def normalize(X, min, max):
    X = (X - min) / (max - min)
    return X

def get_data(file_location):
    data = read_data(file_location)

    train, val, test = trainValTestSplit(data)
    minVal = train.iloc[:, :-1].min()
    maxVal = train.iloc[:, :-1].max()

    X_train = np.array(normalize(train.iloc[:, :-1], minVal, maxVal))
    X_val = np.array(normalize(val.iloc[:, :-1], minVal, maxVal))
    X_test = np.array(normalize(test.iloc[:, :-1], minVal, maxVal))

    y_train = np.array(train.iloc[:, -1:])
    y_val = np.array(val.iloc[:, -1:])
    y_test = np.array(test.iloc[:, -1:])

    return X_train, X_val, X_test, y_train, y_val, y_test


absolutePath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\dataset.csv'
# absolutePath = input('Enter the file location of the dataset: ')
X_train, X_val, X_test, y_train, y_val, y_test = get_data(absolutePath)

print("X_train shape: {}".format(X_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_val shape: {}".format(y_val.shape))
print("y_test shape: {}".format(y_test.shape))

batchSize = [42000, 64, 1]
for size in batchSize:
    print('\nStart training for batchsize =', size, '\n')
    model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=size)
    w, b = model.fit(X_train, y_train)
    y_pred = model.predict(w, b, X_val, 0.5)
    print('\nAcc for batchsize =', size, 'is:', accuracy_score(y_pred, y_val))

