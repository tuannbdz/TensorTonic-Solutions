import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.array(T)
    if type(points[0]) is int:
        points = [points]
    for p in points:
        p.extend([1])
    points = np.array(points)
    p_t = points @ T.T
    p_t = p_t[:, :-1]
    if p_t.shape[0] == 1:
        p_t = p_t.flatten()
    return p_t
    
    