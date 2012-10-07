import numpy as np

def sigmoid_with_noise(n_samples, n_features, outliers=0.,
                       seed=None, noise_amplitude=.2, slope=10.):
    """
    outliers: floating-point in [0, 1]
        fraction of outliers
    """
    # TODO: centering at zero
    np.random.seed(seed)
    w = np.abs(np.random.randn(n_features))
    y_lin = 2. * np.random.rand(n_samples) - 1.
    X = []
    y = []
    for i in range(n_samples):
        #X.append(y_lin[i] * w)
        X.append(y_lin[i] * w)# + noise_amplitude * np.random.randn(n_features))
        y.append(1. / (1. + np.exp(- slope * y_lin[i])))
    y = y + noise_amplitude * np.random.rand(n_samples)

    y = np.array(y) - np.mean(y)
    X = np.array(X)

    for _ in range(int(outliers * n_samples)):
        a = np.random.randint(0, n_samples)
        b = np.random.randint(0, n_samples)
        tmp = y[b]
        y[b] = y[a]
        y[a] = tmp

    return X, y, w