"""
some ordinal regression algorithms
"""
import numpy as np
from scipy import optimize, linalg, stats

def sigma(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def logloss(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out


def obj_margin(x0, X, y, alpha, n_class, weights):
    """
    Objective function for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = np.cumsum(c)
    W = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw # (n_class - 1, n_samples)
    idx = np.arange(n_class - 1)[:, None] < y
    Alpha[idx] *= -1

    return np.sum(W.T * logloss(Alpha)) / float(X.shape[0])


def grad_margin(x0, X, y, alpha, n_class, weights):
    """
    Gradient for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = np.cumsum(c)
    W = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw # (n_class - 1, n_samples)
    idx = np.arange(n_class - 1)[:, None] < y
    Alpha[idx] *= -1
    W[idx.T] *= -1

    Sigma = W.T * sigma(-Alpha)

    grad_w = X.T.dot(Sigma.sum(0)) / float(X.shape[0]) 

    grad_theta = - Sigma.sum(1) / float(X.shape[0])

    tmp = np.concatenate(([0], grad_theta))
    grad_c = np.sum(grad_theta) - np.cumsum(tmp[:-1])

    return np.concatenate((grad_w, grad_c), axis=0)



def threshold_fit(X, y, alpha, n_class, mode='AE', verbose=False):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1'}

    """

    X = np.asarray(X)
    y = np.asarray(y) # XXX check its made of integers
    n_samples, n_features = X.shape

    if mode == 'AE':
        weights = np.ones((n_class, n_class - 1))
    elif mode == '0-1':
        weights = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        weights = np.vstack((weights, np.zeros(n_class -1)))
        weights[-1, -1] = 1



if __name__ == '__main__':

    np.random.seed(1)
    from sklearn import datasets
    n_class = 5
    n_samples = 100

    weights = np.ones((n_class, n_class - 1))

    X, y = datasets.make_regression(n_features=10, noise=20.5)
    bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
    y = np.digitize(y, bins[:-1])
    y -= np.min(y)
    print X.shape
    print y

    x0 = np.random.randn(X.shape[1] + n_class - 1)
    x0[X.shape[1]+1:] = np.abs(x0[X.shape[1]+1:])
    print optimize.approx_fprime(x0, obj_margin, 1e-3, X, y, 0., n_class, weights)
    print grad_margin(x0, X, y, 0., n_class, weights)
