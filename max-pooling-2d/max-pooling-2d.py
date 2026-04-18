import numpy as np
def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    stride = pool_size
    X = np.asarray(X)
    h, w = X.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    out = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            sub_x = X[i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
            out[i, j] = np.max(sub_x)
    return out.tolist()