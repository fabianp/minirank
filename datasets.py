import numpy as np

def sigmoid_with_noise(n_samples, n_features, outliers=0.,
                       seed=None):
    np.random.seed(seed)
    w = np.abs(np.random.randn(n_features))
    y_lin = 2. * np.random.rand(n_samples) - 1.
    X = []
    y = []
    for i in range(n_samples):
        #X.append(y_lin[i] * w)
        X.append(y_lin[i] * w + 0.5 * np.random.rand(n_features))
        y.append(1. / (1. + np.exp(- 5 * y_lin[i])))

    y = np.array(y)
    X = np.array(X)

    for _ in range(int(outliers * n_samples)):
        a = np.random.randint(0, n_samples)
        b = np.random.randint(0, n_samples)
        tmp = y[b]
        y[b] = y[a]
        y[a] = tmp

    order = np.argsort(y)
    X = X[order]
    y = y[order]

    return X, y, w