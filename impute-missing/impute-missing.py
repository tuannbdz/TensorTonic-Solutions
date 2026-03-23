import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    x = np.copy(np.array(X))
    if strategy == 'mean':
        imputed_values = np.nanmean(x, axis=0)
    else:
        imputed_values = np.nanmedian(x, axis=0)
    idx = np.where(np.isnan(x))
    if len(x.shape) != 1:
        x[idx] = np.take(imputed_values, idx[1])
    else:
        x[idx] = imputed_values
    x = np.nan_to_num(x)
    return x
        