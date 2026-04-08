import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import *
from utils import *

# Load data
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(int)

Y = one_hot(y)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Transpose
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

# Training
def train_model(X, Y, epochs=1000, lr=0.1):
    W1, b1, W2, b2 = init_params()
    losses = []

    for i in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = compute_loss(A2, Y)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, A2, W2)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return W1, b1, W2, b2, losses

# Run training
W1, b1, W2, b2, losses = train_model(X_train, Y_train)

# Evaluate
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)

preds = predict(X_test, W1, b1, W2, b2)
print("Test Accuracy:", accuracy(preds, Y_test))

# Plot loss
plt.plot(losses)
plt.title("Training Loss")
plt.savefig("results/loss.png")
plt.show()