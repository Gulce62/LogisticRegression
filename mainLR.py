import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import LogisticRegression as LR


def read_data(file_name):
    # Read the data with its path location
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception:
        sys.exit(1)


def trainValTestSplit(data):
    # Split the data into train, validation, test
    shuffled = data.sample(frac=1, random_state=1)
    dataSize = len(shuffled)
    train = shuffled[:int(dataSize * 0.7)]
    val = shuffled[int(dataSize * 0.7):int(dataSize * 0.8)]
    test = shuffled[int(dataSize * 0.8):]
    return train, val, test


def normalize(X, min, max):
    # Normalize the features
    X = (X - min) / (max - min)
    return X


def get_data(file_location):
    # Get X_train, X_val, X_test, y_train, y_val, y_test data
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


def getConfusionMatrix(y_true, y_pred):
    plt.figure()
    # Get the confusion matrix
    y_true = pd.Categorical(y_true.ravel())
    y_pred = pd.Categorical(y_pred.ravel())
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True)
    return confusion_matrix


def getValAccPlot(val_acc, paramaterName):
    # Get the accuracy plot for every epoch with validation set
    title = 'Validation Accuracy For Every Epoch With Different ' + paramaterName
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.plot(val_acc)


absolutePath = input('Enter the file location of the dataset: ')
X_train, X_val, X_test, y_train, y_val, y_test = get_data(absolutePath)

print("X_train shape: {}".format(X_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_val shape: {}".format(y_val.shape))
print("y_test shape: {}".format(y_test.shape))


# Find the optimal batch size
batchSize = [42000, 64, 1]
batchSizeAcc = []
parameterList = []
plt.figure()
for size in batchSize:
    print('\nStart training for batch size =', size, '\n')
    model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=size, initialization='gaussian')
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    parameterList.append((w, b))
    getValAccPlot(val_acc, 'Batch Size')
    y_pred = model.predict(w, b, X_val, 0.5)
    accuracy = LR.getAccuracy(y_pred, y_val)
    batchSizeAcc.append(accuracy)
    print('\nFinal accuracy for batch size =', size, 'is:', accuracy)
plt.legend(['full-batch: 42000', 'mini-batch: 64', 'stochastic: 1'])

maxBatchSize = batchSize[np.argmax(batchSizeAcc)]
model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=maxBatchSize, initialization='gaussian')
w, b = parameterList[np.argmax(batchSizeAcc)]
y_pred = model.predict(w, b, X_test, 0.5)
getConfusionMatrix(y_pred, y_test)
title = 'Confusion Matrix for batch size = ' + str(maxBatchSize)
plt.title(title)

# Find the optimal initialization
initializationType = ['gaussian', 'uniform', 'zero']
initializationAcc = []
parameterList = []
plt.figure()
for inType in initializationType:
    print('\nStart training for', inType, 'initialization\n')
    model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=64, initialization=inType)
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    parameterList.append((w, b))
    getValAccPlot(val_acc, 'Initialization Type')
    y_pred = model.predict(w, b, X_val, 0.5)
    accuracy = LR.getAccuracy(y_pred, y_val)
    initializationAcc.append(accuracy)
    print('\nFinal accuracy for for', inType, 'initialization is:', accuracy)
plt.legend(initializationType)

maxInitialization = initializationType[np.argmax(initializationAcc)]
model = LR.LogisticRegression(epochs=100, learningRate=0.001, batchSize=64, initialization=maxInitialization)
w, b = parameterList[np.argmax(initializationAcc)]
y_pred = model.predict(w, b, X_test, 0.5)
getConfusionMatrix(y_pred, y_test)
title = 'Confusion Matrix for ' + str(maxInitialization) + ' initialization'
plt.title(title)

# Find the optimal learning rate
learningRate = [1, 0.001, 0.0001, 0.00001]
learningRateAcc = []
parameterList = []
plt.figure()
for rate in learningRate:
    print('\nStart training for learning rate =', rate, '\n')
    model = LR.LogisticRegression(epochs=100, learningRate=rate, batchSize=64, initialization='gaussian')
    w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
    parameterList.append((w, b))
    getValAccPlot(val_acc, 'Learning Rate')
    y_pred = model.predict(w, b, X_val, 0.5)
    accuracy = LR.getAccuracy(y_pred, y_val)
    learningRateAcc.append(accuracy)
    print('\nFinal accuracy for learning rate =', rate, 'is:', accuracy)
plt.legend(learningRate)

maxLearningRate = learningRate[np.argmax(learningRateAcc)]
model = LR.LogisticRegression(epochs=100, learningRate=maxLearningRate, batchSize=64, initialization='gaussian')
w, b = parameterList[np.argmax(learningRateAcc)]
y_pred = model.predict(w, b, X_test, 0.5)
getConfusionMatrix(y_pred, y_test)
title = 'Confusion Matrix for learning rate = ' + str(maxLearningRate)
plt.title(title)

# Final training with the best hyperparameters
print('\nStart final training with best parameters\n')
print('epochs=100, learningRate=', maxLearningRate, 'batchSize=', maxBatchSize, 'initialization=', maxInitialization)
model = LR.LogisticRegression(epochs=100, learningRate=maxLearningRate, batchSize=maxBatchSize,
                              initialization=maxInitialization)
w, b, val_acc = model.fit(X_train, y_train, X_val, y_val)
y_pred = model.predict(w, b, X_test, 0.5)
accuracy = LR.getAccuracy(y_pred, y_test)
cm = getConfusionMatrix(y_pred, y_test)
plt.title('Confusion Matrix For Test Set')
plt.show()

# Calculate metrics
true_pos = np.diag(cm)[0]
false_pos = np.sum(cm, axis=1)[0] - true_pos

pre_denom = np.sum(cm, axis=1)[0]
rec_denom = np.sum(cm, axis=0)[0]
FPR_denom = np.sum(cm, axis=0)[1]

precision = true_pos/pre_denom
recall = true_pos/rec_denom
FPR = false_pos/FPR_denom
F1_beta_square = 1**2
F1_measure = ((1 + F1_beta_square)*precision*recall)/((F1_beta_square*precision)+recall)
F2_beta_square = 2**2
F2_measure = ((1 + F2_beta_square)*precision*recall)/((F2_beta_square*precision)+recall)
F05_beta_square = 0.5**2
F05_measure = ((1 + F05_beta_square)*precision*recall)/((F05_beta_square*precision)+recall)

print('\nAccuracy for test set is:', accuracy)
print('Precision for test set is:', precision)
print('Recall for test set is:', recall)
print('False Positive Rate (FPR) for test set is:', FPR)
print('F1 measure for test set is:', F1_measure)
print('F2 measure for test set is:', F2_measure)
print('F0.5 measure for test set is:', F05_measure)