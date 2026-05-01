import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Your code here
    points = np.asarray(points)
    R = np.asarray([[np.cos(theta), -np.sin(theta),  0], \
                    [np.sin(theta),  np.cos(theta),  0], \
                    [0, 0, 1]
                   ]
                  )
    if(len(points.shape) == 1):
        points = points[np.newaxis, :]
        return (R @ points.T).flatten()
    return (R @ points.T).T