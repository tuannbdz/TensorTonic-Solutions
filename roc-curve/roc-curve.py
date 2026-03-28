import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc_idx = np.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tpr = []
    fpr = []
    thresholds = []
    thresholds.append('inf')

    tp = 0
    fp = 0

    prev_score = None

    for i in range(len(y_score)):
        if prev_score is None or y_score[i] != prev_score:
            tpr.append(tp / P if P > 0 else 0)
            fpr.append(fp / N if N > 0 else 0)
            thresholds.append(y_score[i])
            prev_score = y_score[i]

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(1.0)
    fpr.append(1.0)

    return np.array(fpr), np.array(tpr), np.array(thresholds)