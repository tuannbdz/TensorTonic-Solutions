import numpy as np
def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    n = len(series)
    muy = np.mean(series)
    var = np.var(series) * n
    lag = []
    if var == 0:
        return [1] + [0] * (max_lag)
    for k in range(max_lag + 1):
        r_k = 0
        for t in range(n - k):
            r_k += (series[t] - muy) * (series[t + k] - muy)
        r_k /= var
        lag.append(r_k)
    return lag