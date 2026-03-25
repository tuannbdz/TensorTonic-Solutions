import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    Returns: n x k list of floats
    """
    X = np.array(X)
    n, d = X.shape
    
    # 1. Center the data
    X_c = X - np.mean(X, axis=0, keepdims=True)
    
    # 2. Compute the sample covariance matrix (n-1)
    # Note: Use X_c.T @ X_c
    C = (X_c.T @ X_c) / (n - 1)
    
    # 3. Find eigenvalues and eigenvectors
    # eigh is optimized for symmetric matrices like covariance
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # 4. Find the top-k eigenvectors (ordered by decreasing eigenvalue)
    # np.argsort returns indices to sort ascending; we take the last k and reverse
    idx = np.argsort(eigenvalues)[::-1][:k]
    top_k_ev = eigenvectors[:, idx]
    
    # 5. Project the centered data
    # Resulting shape: (n, d) @ (d, k) = (n, k)
    projection = X_c @ top_k_ev
    
    # 6. Return as list of floats
    return projection.tolist()