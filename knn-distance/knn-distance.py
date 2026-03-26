import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    n, m = len(X_train), len(X_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    X_train = X_train[:, np.newaxis]
    X_test = X_test[np.newaxis, :]
    diff = X_train - X_test
    dist = np.linalg.norm(diff, axis=-1)
    knn = np.argsort(dist, axis=0)[:k, :].T
    if k > n:
        knn = np.pad(knn, (0,k - n), constant_values=-1)[:m, :]
    return knn