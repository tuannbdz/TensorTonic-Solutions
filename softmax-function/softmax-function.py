import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    z = x - np.max(x, axis=1, keepdims=True)
    e_z = np.exp(z)
    ans = e_z / np.sum(e_z, axis=1, keepdims=True)
    return ans[0] if x.shape[0] == 1 else ans
        