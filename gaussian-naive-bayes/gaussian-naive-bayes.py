import numpy as np
def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    n = len(y_train)
    n_cls, cnt_cls = np.unique(y_train, return_counts=True)
    prop = []
    for c in range(len(n_cls)):
        cls_idx = np.where(y_train == c)[0]
        X_cls = X_train[cls_idx]
        prior_p = cnt_cls[c] / n
        mean_c = np.mean(X_cls, axis=0, keepdims=True)
        var_c = np.var(X_cls, axis=0, keepdims=True) + 1e-9
        gauss_est = np.sum(-0.5 * np.log(2*np.pi*var_c) - np.square(X_test - mean_c)/ (2 * var_c), axis=-1, keepdims=True)
        prop_c = np.log(prior_p) + gauss_est
        prop.append(prop_c)
        # print(prior_p, gauss_est, mean_c, var_c, prop_c)
    prop = np.concatenate(prop, axis=-1)
    return list(np.argmax(prop, axis=-1))
        