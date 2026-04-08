import numpy as np

def one_hot(y):
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

def accuracy(preds, Y):
    labels = np.argmax(Y, axis=0)
    return np.mean(preds == labels)