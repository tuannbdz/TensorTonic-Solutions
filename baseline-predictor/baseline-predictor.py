import numpy as np
def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    ratings_matrix = np.asarray(ratings_matrix, dtype=float)
    target_pairs = np.asarray(target_pairs)
    ratings_matrix[ratings_matrix == 0] = np.nan
    ga = np.nanmean(ratings_matrix)
    users_b = np.nanmean(ratings_matrix, axis=1)
    items_b = np.nanmean(ratings_matrix, axis=0)
    ans = []
    for pair in target_pairs:
        user_b = users_b[pair[0]] - ga 
        item_b = items_b[pair[1]] - ga
        ans.append(ga + user_b + item_b)
    return ans