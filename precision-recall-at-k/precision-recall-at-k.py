def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    topk = set(recommended[:k])
    cnt = 0
    for i in relevant:
        if i in topk:
            cnt += 1
    return [cnt / k, cnt / len(relevant)]