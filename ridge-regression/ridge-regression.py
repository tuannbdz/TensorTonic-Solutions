def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    t0 = (X.T @ X)
    t0 = t0 + lam * np.eye(len(t0))
    return (np.linalg.inv(t0) @ X.T @ y).flatten()