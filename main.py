import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import sklearn.datasets
import sklearn.linear_model as l
import LogisticRegression as LR


def read_data(file_name):
    # Read the data with its path location
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception:
        sys.exit(1)


def trainValTestSplit(data):
    shuffled = data.sample(frac=1, random_state=1)
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
    y_val = np.array(val.iloc[:, -1:]).ravel()
    y_test = np.array(test.iloc[:, -1:]).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test


absolutePath = r'C:\Users\gulce\Desktop\EEE 8TH SEMESTER\CS 464\Homeworks\HW2\dataset.csv'
# absolutePath = input('Enter the file location of the dataset: ')
X_train, X_val, X_test, y_train, y_val, y_test = get_data(absolutePath)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("y_val shape: {}".format(y_val.shape))

"""
clf = l.LogisticRegression()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)"""
ones = np.ones((1, X_train.shape[0]))
X_train = np.insert(X_train, 0, ones, axis=1)
print(X_train.shape)
print(X_train)

model = LR.LogisticRegression(10, 1)
w = model.fit(X_train, y_train)
Y_prediction_train = model.predict(w, X_train)

print(Y_prediction_train)
accuracyBool = (Y_prediction_train == y_train)
print('Accuracy: ', np.count_nonzero(accuracyBool) / accuracyBool.shape[0])
