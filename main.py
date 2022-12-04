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


def getConfusionMatrix(y_true, y_pred, axs):
    y_true = pd.Categorical(y_true.ravel())
    y_pred = pd.Categorical(y_pred.ravel())
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True, ax=axs)


def getValAccPlot(val_acc, paramaterName):
    title = 'Validation Accuracy For Every Epoch With Different ' + paramaterName
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.plot(val_acc)


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
batchSizeAcc = []
y_predList = []
plt.figure()
for size in batchSize:
    print('\nStart training for batch size =', size, '\n')
    model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=size, initialization='gaussian')
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    getValAccPlot(val_acc, 'Batch Size')
    y_pred = model.predict(w, b, X_val, 0.5)
    y_predList.append(y_pred)
    accuracy = LR.getAccuracy(y_pred, y_val)
    batchSizeAcc.append(accuracy)
    print('\nFinal accuracy for batch size =', size, 'is:', accuracy)
plt.legend(str(batchSize))
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(16, 16))
for i in range(0, 3):
    title = 'Confusion Matrix for batch size =' + str(batchSize[i])
    plt.title(title)
    getConfusionMatrix(y_predList[i], y_val, axs[i])
plt.show()

initializationType = ['gaussian', 'uniform', 'zero']
initializationAcc = []
y_predList = []
plt.figure()
for inType in initializationType:
    print('\nStart training for', inType, 'initialization\n')
    model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=64, initialization=inType)
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    getValAccPlot(val_acc, 'Initialization Type')
    y_pred = model.predict(w, b, X_val, 0.5)
    y_predList.append(y_pred)
    accuracy = LR.getAccuracy(y_pred, y_val)
    initializationAcc.append(accuracy)
    print('\nFinal accuracy for for', inType, 'initialization is:', accuracy)
plt.legend(str(initializationType))
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(32, 8))
for i in range(0, 3):
    title = 'Confusion Matrix for' + str(initializationType[i]) + 'initialization'
    plt.title(title)
    getConfusionMatrix(y_predList[i], y_val, axs[i])
plt.show()

learningRate = [0.001, 0.0001, 0.00001]
learningRateAcc = []
y_predList = []
plt.figure()
for rate in learningRate:
    print('\nStart training for learning rate =', rate, '\n')
    model = LR.LogisticRegression(epochs=100, learningRate=rate, batchSize=64, initialization='gaussian')
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    getValAccPlot(val_acc, 'Learning Rate')
    y_pred = model.predict(w, b, X_val, 0.5)
    y_predList.append(y_pred)
    accuracy = LR.getAccuracy(y_pred, y_val)
    learningRateAcc.append(accuracy)
    print('\nFinal accuracy for learning rate =', rate, 'is:', accuracy)
plt.legend(str(learningRate))
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(32, 8))
for i in range(0, 3):
    title = 'Confusion Matrix for learning rate =' + str(learningRate[i])
    plt.title(title)
    getConfusionMatrix(y_predList[i], y_val, axs[i])
plt.show()

print('\nStart final training with best parameters\n')
maxBatchSize = batchSize[np.argmax(batchSizeAcc)]
maxInitialization = initializationType[np.argmax(initializationAcc)]
maxLearningRate = learningRate[np.argmax(learningRateAcc)]
model = LR.LogisticRegression(epochs=100, learningRate=maxLearningRate, batchSize=maxBatchSize,
                              initialization=maxInitialization)
w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
y_pred = model.predict(w, b, X_test, 0.5)
accuracy = LR.getAccuracy(y_pred, y_test)
print('\nAccuracy for test set is:', accuracy)
fig, axs = plt.subplots(1, 1, figsize=(32, 8))
getConfusionMatrix(y_pred, y_test, axs)
