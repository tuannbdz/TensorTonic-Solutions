import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    predictions = np.asarray(predictions)
    targets = np.array(targets)
    predictions = predictions * targets + (1 - predictions) * (1 - targets)
    return np.mean(-alpha * np.pow(1 - predictions, gamma) * np.log(predictions))