import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions)
    ans = []
    for i in range(predictions.shape[1]):
        x, counts = np.unique(predictions[:, i], return_counts=True)
        ans.append(x[np.argmax(counts)])
    return ans