import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    values = np.array(values)
    transitions = np.array(transitions)
    rewards = np.array(rewards)
    n_s, n_actions, n_ss = transitions.shape
    ans = []
    for s in range(n_s):
        max_reward = 0
        for a in range(n_actions):
            reward = rewards[s][a] + gamma * np.dot(transitions[s, a, :], values)
            max_reward = max(max_reward, reward)
        ans.append(max_reward)
    return ans