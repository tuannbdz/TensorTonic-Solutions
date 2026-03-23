import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    n_cls, idx_cls = np.unique(y, return_inverse=True)
    idx_train, idx_test = [], []
    for cls in range(len(n_cls)):
        idx_v_cls = np.where((y == cls))[0]
        cutoff = round(len(idx_v_cls) *  test_size)
        if rng != None:
            rng.shuffle(idx_v_cls)
        else:
            np.random.shuffle(idx_v_cls)
        idx_test.extend(sorted(idx_v_cls[:cutoff]))
        idx_train.extend(sorted(idx_v_cls[cutoff:]))
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]
    
    