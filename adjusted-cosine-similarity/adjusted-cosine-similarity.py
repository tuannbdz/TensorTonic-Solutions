import numpy as np
def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    ratings_matrix = np.asarray(ratings_matrix)
    mask = (
        (ratings_matrix[:, item_i] != 0) &
        (ratings_matrix[:, item_j] != 0)
    )

    if np.sum(mask) == 0:
        return 0
    users = ratings_matrix[mask]
    user_means = np.true_divide(
        users.sum(axis=1),
        (users != 0).sum(axis=1)
    )
    vec_i = users[:, item_i] - user_means
    vec_j = users[:, item_j] - user_means

    denom = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)

    if denom == 0:
        return 0

    return np.dot(vec_i, vec_j) / denom
