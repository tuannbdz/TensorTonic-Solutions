import numpy as np

def gini_impurity(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X = np.array(X)
    y= np.array(y)
    n_samples, n_features = X.shape
    parent_impurity = gini_impurity(y)

    best_feature = None
    best_threshold = None
    best_gain = -1

    for feature in range(n_features):
        # Sort data along this feature
        sorted_idx = np.argsort(X[:, feature])
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]

        for i in range(1, n_samples):
            if X_sorted[i, feature] == X_sorted[i - 1, feature]:
                continue

            threshold = (X_sorted[i, feature] + X_sorted[i - 1, feature]) / 2

            left_y = y_sorted[:i]
            right_y = y_sorted[i:]

            left_impurity = gini_impurity(left_y)
            right_impurity = gini_impurity(right_y)

            left_weight = len(left_y) / n_samples
            right_weight = len(right_y) / n_samples

            gain = parent_impurity - (
                left_weight * left_impurity + right_weight * right_impurity
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold