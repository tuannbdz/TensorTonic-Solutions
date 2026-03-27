import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x)
    dim = len(x.shape)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if dim == 2:
        muy = np.mean(x, axis=0)
        var = np.var(x, axis=0)
    else:
        muy = np.mean(x, axis=(0, 2, 3))
        var = np.var(x, axis=(0, 2, 3))
        muy = muy[np.newaxis, :, np.newaxis, np.newaxis]
        var = var[np.newaxis, :, np.newaxis, np.newaxis]
        gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]
        beta = beta[np.newaxis, :, np.newaxis, np.newaxis]
    norm_x = (x - muy) / np.sqrt(var + eps)
    return norm_x * gamma + beta