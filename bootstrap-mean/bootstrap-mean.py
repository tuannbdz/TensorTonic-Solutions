import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    alpha = (1 - ci) / 2
    x = np.asarray(x)
    N = len(x)
    if rng == None:
        rng = np.random.default_rng()
    # else:
        # rng = np.random.default_rng(rng)
    boot = x[rng.integers(N, size=(n_bootstrap, N))]
    boot_mean = np.mean(boot, axis=-1)
    lower, upper = np.quantile(boot_mean, alpha), np.quantile(boot_mean, 1 - alpha)
    return (boot_mean, lower, upper)
