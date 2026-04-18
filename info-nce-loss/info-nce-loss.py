import numpy as np
def softmax(X):
    z = np.exp(X - np.max(X))
    return z / np.sum(z, axis=1, keepdims=True)

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    N = Z1.shape[0]
    S = Z1 @ Z2.T / temperature
    L = - np.sum(np.log(np.linalg.diagonal(softmax(S)))) / N
    return L