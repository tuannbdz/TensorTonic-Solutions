import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    numerator = np.dot(v, w)
    denominator = np.linalg.norm(v) * np.linalg.norm(w)
    return np.arccos(numerator / denominator)