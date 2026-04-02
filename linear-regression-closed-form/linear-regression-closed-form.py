import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    theta = np.linalg.inv(X.T @ X)  @ X.T @ y
    return theta.flatten()
