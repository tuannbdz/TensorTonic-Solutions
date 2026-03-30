import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    n = len(y)
    _,counts = np.unique(y, return_counts=True)
    p = counts / n
    return -np.sum(p * np.log2(p))