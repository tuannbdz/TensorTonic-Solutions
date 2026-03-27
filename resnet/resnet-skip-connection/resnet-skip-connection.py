import numpy as np


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    grad = np.eye(gradients_F[0].shape[0])
    for J in gradients_F:
        grad = J @ grad
    x = np.asarray(x).reshape(-1)
    return grad @ x


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    """
    dim = gradients_F[0].shape[0]
    grad = np.eye(dim)
    
    for J in gradients_F:
        grad = (np.eye(dim) + J) @ grad

    x = np.asarray(x).reshape(-1)
    
    return grad @ x

    