import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    n = len(x.shape)
    if n < 3 or n > 4:
        raise ValueError("ValueError")
    return np.mean(x, axis=(-1, -2))