import numpy as np

def dropout(x:np.ndarray, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    rand = rng.random if rng is not None else np.random.random
    keep_mask = rand(x.shape) >= p
    scale = 1.0 / (1.0 - p)
    dropout_pattern = keep_mask.astype(x.dtype) * scale
    output = x * dropout_pattern

    return output, dropout_pattern
    
    