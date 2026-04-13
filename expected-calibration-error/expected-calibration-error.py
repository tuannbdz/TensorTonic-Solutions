import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])

        if np.any(mask):
            bin_size = np.sum(mask)
            avg_confidence = np.mean(y_pred[mask])
            avg_accuracy = np.mean(y_true[mask])

            ece += (bin_size / N) * np.abs(avg_confidence - avg_accuracy)

    return ece