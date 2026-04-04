def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob_distributions = np.array(prob_distributions)
    actual_tokens = np.array(actual_tokens)
    N = len(actual_tokens)
    idx = np.arange(N)
    p = prob_distributions[idx, actual_tokens]
    return np.exp(-np.sum(np.log(p)) / N)