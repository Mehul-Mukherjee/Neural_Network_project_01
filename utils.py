import numpy as np

# -------------------------
# Mini-batch generator
# -------------------------
def get_batches(X, Y, batch_size=64):
    m = X.shape[1]

    # shuffle data
    indices = np.random.permutation(m)
    X = X[:, indices]
    Y = Y[:, indices]

    batches = []

    for i in range(0, m, batch_size):
        X_batch = X[:, i:i + batch_size]
        Y_batch = Y[:, i:i + batch_size]
        batches.append((X_batch, Y_batch))

    return batches


# -------------------------
# Accuracy function
# -------------------------
def accuracy(preds, Y):
    labels = np.argmax(Y, axis=0)
    return np.mean(preds == labels)