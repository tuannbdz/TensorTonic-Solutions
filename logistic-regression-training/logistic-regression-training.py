import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    w = np.zeros(X.shape[-1])
    b = np.zeros(1)
    for epoch in range(steps):
        logits = np.sum(X * w, axis=-1, keepdims=True) + b
        pred = _sigmoid(logits)
        loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(pred))
        dl_dw = (pred - y) * X
        dl_db = (pred - y)

        w -= lr * np.sum(dl_dw)
        b -= lr * np.sum(dl_db)
        
    return w, b