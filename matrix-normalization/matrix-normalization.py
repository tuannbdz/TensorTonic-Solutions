import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    norm = 0
    matrix = np.asarray(matrix)
    if len(matrix.shape) != 2:
       return None
    try:
        if norm_type == 'l2':
            norm = np.sqrt(np.sum(np.power(matrix, 2), axis=axis, keepdims=True))
        elif norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'max':
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None
    except:
        return None
    if axis == 1:
        ans = matrix / norm.reshape(-1, 1)
    elif axis == 0:
        ans = matrix / norm.reshape(1, -1)
    elif axis == None:
        ans = matrix / norm
    else:
        return None
    ans = np.nan_to_num(ans)
    return ans