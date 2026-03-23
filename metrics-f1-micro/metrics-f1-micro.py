def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    tp = 0
    for a, b in zip(y_true, y_pred):
        if(a == b):
            tp += 1
    fp_fn = len(y_true) - tp
    return tp / (tp + fp_fn)