def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_a = set(set_a)
    set_b = set(set_b)
    try:
        return len(set_a & set_b) / len(set_a | set_b)
    except:
        return 0