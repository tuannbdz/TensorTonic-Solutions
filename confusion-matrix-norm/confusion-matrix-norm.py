import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    if normalize == 'none':
        return cm
    elif normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sums, where=row_sums != 0)
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, col_sums, where=col_sums != 0)
    elif normalize == 'all':
        total = cm.sum()
        return cm / total if total != 0 else cm