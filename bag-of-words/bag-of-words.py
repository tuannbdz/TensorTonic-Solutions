import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    N = len(vocab)
    vocab_d = {}
    for i, word in enumerate(vocab):
        vocab_d[word] = 0
    for word in tokens:
        if word in vocab_d:
            vocab_d[word] += 1
    return np.asarray([v for k, v in vocab_d.items()], dtype=int)
    