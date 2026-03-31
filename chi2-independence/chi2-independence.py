import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.array(C)
    N = np.sum(C)
    E = np.sum(C, axis=1).reshape(-1, 1) * np.sum(C, axis=0).reshape(1, -1)
    E = E / N
    chi2 = np.sum((C - E)**2 / E)
    return chi2, E
    