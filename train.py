import numpy as np
import matplotlib.pyplot as plt

from model import (
    init_params,
    forward,
    backward,
    update,
    compute_loss,
    predict
)

from utils import get_batches, accuracy


# -------------------------
# TRAINING FUNCTION
# -------------------------
def train(X, Y, epochs=20, lr=0.1, batch_size=64):

    W1, b1, W2, b2 = init_params()

    losses = []
    accs = []

    for epoch in range(epochs):

        batches = get_batches(X, Y, batch_size)

        epoch_loss = 0

        for X_batch, Y_batch in batches:

            # forward
            Z1, A1, Z2, A2 = forward(X_batch, W1, b1, W2, b2)

            # loss
            loss = compute_loss(A2, Y_batch)
            epoch_loss += loss

            # backward
            dW1, db1, dW2, db2 = backward(
                X_batch, Y_batch, Z1, A1, A2, W2
            )

            # update
            W1, b1, W2, b2 = update(
                W1, b1, W2, b2,
                dW1, db1, dW2, db2,
                lr
            )

        # end of epoch metrics
        avg_loss = epoch_loss / len(batches)
        losses.append(avg_loss)

        preds = predict(X, W1, b1, W2, b2)
        acc = accuracy(preds, Y)
        accs.append(acc)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    return W1, b1, W2, b2, losses, accs


# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1)

    X = mnist.data.to_numpy().astype(np.float32) / 255.0
    y = mnist.target.to_numpy().astype(int)

    # one-hot encoding
    def one_hot(y):
        one_hot_y = np.zeros((y.size, 10))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y

    Y = one_hot(y)

    # split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # transpose for NN
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    # train
    W1, b1, W2, b2, losses, accs = train(X_train, Y_train)

    # test accuracy
    preds = predict(X_test, W1, b1, W2, b2)
    test_acc = accuracy(preds, Y_test)

    print("\nFinal Test Accuracy:", test_acc)

    # -------------------------
    # PLOT RESULTS
    # -------------------------
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/loss_curve.png")
    plt.show()

    plt.plot(accs)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("results/accuracy_curve.png")
    plt.show()