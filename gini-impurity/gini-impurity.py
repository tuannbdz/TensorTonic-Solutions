import numpy as np

def gini(t):
    if len(t) == 0:
        return 0.0
    _, counts = np.unique(t, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.dot(probs, probs)


def gini_impurity(y_left, y_right):
    N_l = len(y_left)
    N_r = len(y_right)
    N = N_l + N_r

    if N == 0:
        return 0.0

    return (N_l * gini(y_left) + N_r * gini(y_right)) / N