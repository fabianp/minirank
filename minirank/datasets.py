import numpy as np

def sigmoid_with_noise(n_samples, n_features, outliers=0.,
                       seed=None, noise_amplitude=.2, slope=1.):
    """
    outliers: floating-point in [0, 1]
        fraction of outliers
    """
    np.random.seed(seed)
    w = np.random.rand(n_features)
    #w /= np.linalg.norm(w)
    y = 2 * (np.random.rand(n_samples) - 0.5)
    p = y[:, np.newaxis] * w

    # centering at [0, 1]
    p -= np.min(p)
    p /= np.max(p)

    #p /= 100

    # p lies in [0, 1] but we'd like it to lie in [epsilon, 1 - epsilon]
    # with epsilon a small quantity
    epsilon = 1e-6 / 2
    m = 1 / (1 + epsilon)
    p *= m
    p += (1 - m) / 2
    #assert np.abs(p.min() - (1 - p.max())) < 1e-3
    X = np.log(p) - np.log(1 - p)  + noise_amplitude * np.random.rand(n_samples, n_features)

    # normalize features
#    for i in range(X.shape[1]):
#        X[:, i] /= np.linalg.norm(X[:, i])

    for _ in range(int(outliers * n_samples)):
        a = np.random.randint(0, n_samples)
        b = np.random.randint(0, n_samples)
        tmp = y[b]
        y[b] = y[a]
        y[a] = tmp

    return X, y, w