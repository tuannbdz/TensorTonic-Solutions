import numpy as np
def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    novelty_score = 0
    for r in recommendations:
        novelty_score += -np.log2(item_counts[r] / n_users)
    novelty_score /= len(recommendations)
    return novelty_score