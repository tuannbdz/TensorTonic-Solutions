import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.

    Args:
        log_probs: array-like of shape (T,) — log π(a_t | s_t)
        rewards: array-like of shape (T,)
        gamma: discount factor

    Returns:
        scalar loss (float)
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    baseline = returns.mean()
    advantages = returns - baseline
    loss = -np.mean(log_probs * advantages)
    return float(loss)