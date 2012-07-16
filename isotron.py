"""
The Isotron algorithm

Reference: "The Isotron Algorithm: High-Dimensional Isotonic Regression",
    Adam Tauman Kalai, Ravi Sastry
"""

import numpy as np
from isotonic_regression import pav

def isotron(X, y, max_iter=100):
    """
    Parameters
    ----------
    X : array, shape (n_features, n_samples)
    y : array, shape (n_samples,)

    Returns
    -------
    w : coefficients of the model
    u : function u evaluated at Xw
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    u = y.copy()
    for i in range(max_iter):
        idx = np.argsort(np.dot(X, w))
        u = pav(y[idx])
        w += (1. / n_samples) * np.sum((y[idx] - u) * X[idx].T, 1)
    order_inv = np.zeros(len(y), dtype=np.int)
    order_inv[idx] = np.arange(len(y))
    return w, u[order_inv]

if __name__ == '__main__':
    import pylab as pl
    n_samples = 300
    n_features = 25

    np.random.seed(0)
    rng = np.random.RandomState(42)
    w = np.abs(rng.randn(n_features))
    y_lin = 2 * np.random.rand(n_samples) - 1
    X = []
    y = []
    for i in range(n_samples):
        X.append(y_lin[i] * w + rng.rand(n_features))
        y.append(1. / (1. + np.exp(- 5 * y_lin[i])))

    y = np.array(y)
    X = np.array(X)

    w1, u1 = isotron(X, y, max_iter=100)
    print "corr_coef : ", np.corrcoef(w, w1)[0, 1]

    z = np.dot(X, w)
    z1 = np.dot(X, w1)
    pl.scatter(z, y)
    pl.scatter(z, u1, color='red')
    pl.show()

