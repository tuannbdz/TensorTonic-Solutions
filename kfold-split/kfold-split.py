import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    indices = np.arange(N)
    if shuffle:
        if rng is not None:
            indices = rng.permutation(N)
        else:
            indices = indices.copy()
            np.random.shuffle(indices)
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[:N % k] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx.astype(int), val_idx.astype(int)))
        current = end
    return folds
