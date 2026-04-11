import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [v if v > 0 else alpha * (np.exp(v) - 1) for v in x]