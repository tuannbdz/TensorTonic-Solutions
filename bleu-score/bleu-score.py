import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c = len(candidate)
    r = len(reference)
    if c == 0:
        return 0.0
    
    BP = 1 if c >= r else np.exp(1 - r / c)
    candidate = np.array(candidate)
    reference = np.array(reference)

    p = []
    for pi in range(1, max_n+1):
        cp = sliding_window_view(candidate, pi)
        rp = sliding_window_view(reference, pi)
        s = 0
        for i in range(min(len(cp), len(rp))):
            if ''.join(cp[i]) == ''.join(rp[i]):
                s += 1
        s /= len(cp)
        p.append(s)
    p = np.array(p)
    return BP * np.exp(np.mean(np.log(p)))
        
    