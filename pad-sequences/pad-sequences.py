import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = 0
    for seq in seqs:
        N = max(N, len(seq))
    if max_len == None:
        for seq in seqs:
            seq.extend([pad_value] * (N - len(seq)))
    else:
        for seq in seqs:
            if len(seq) >= max_len:
                del seq[max_len:]
            else:
                seq.extend([pad_value] * (max_len - len(seq)))
    return np.array(seqs)
        