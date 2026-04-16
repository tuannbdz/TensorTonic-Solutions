import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y)
    y_l = y[split_mask]
    y_r = y[~split_mask]
    N = len(y)
    n_l, n_r = len(y_l), len(y_r)
    return _entropy(y) - (n_l * _entropy(y_l) + n_r * _entropy(y_r)) / N
