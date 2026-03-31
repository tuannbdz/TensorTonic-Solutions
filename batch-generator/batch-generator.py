import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    N = len(X)
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.permutation(N)

    for start in range(0, N, batch_size):
        end = start + batch_size

        if end > N and drop_last:
            break

        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]