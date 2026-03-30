import numpy as np

def expected_value_discrete(x: np.ndarray, p: np.ndarray):
    """
    Returns: float expected value
    """
    # Write code here
    if np.sum(p) != 1:
        raise ValueError('ValueError')
    else:
        return np.dot(x, p)
