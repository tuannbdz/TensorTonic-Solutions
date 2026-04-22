import numpy as np
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    return np.log(1 + np.asarray(values))
