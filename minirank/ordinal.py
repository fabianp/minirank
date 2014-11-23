"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize, linalg, stats

from sklearn import base


def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def log_loss(Z):
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

    return np.sum(W.T * log_loss(Alpha)) / float(X.shape[0]) + \
           alpha * (linalg.norm(w) ** 2)


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

    Sigma = W.T * sigmoid(-Alpha)

    grad_w = X.T.dot(Sigma.sum(0)) / float(X.shape[0]) + alpha * 2 * w

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

    x0 = np.zeros(n_features + n_class - 1)
    bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_class - 2)
    sol = optimize.minimize(obj_margin, x0,
        args=(X, y, alpha, n_class, weights), method='L-BFGS-B', bounds=bounds)
    if not sol.success:
        print(sol.status)
    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = np.cumsum(c)
    return w, theta


def threshold_predict(X, w, theta):
    """
    Class numbers are between 0 and k-1
    """
    pred = []
    n_samples = X.shape[0]
    Xw = X.dot(w)
    tmp = theta[:, None] - Xw
    return np.sum(tmp <= 0, axis=0).astype(np.int)


class MarginOR(base.BaseEstimator):
    def __init__(self, alpha, n_class, mode='AE', scoring='AE', verbose=0):
        self.alpha = alpha
        self.mode = mode
        self.scoring = scoring
        self.n_class = n_class
        self.verbose = verbose

    def fit(self, X, y):
        self.w_, self.theta_ = threshold_fit(X, y, self.alpha, n_class,
            mode=self.mode, verbose=self.verbose)
        return self

    def pred(self, X):
        return threshold_predict(X, self.w_, self.theta_)

    def score(self, X, y):
        pred = self.pred(X)
        if self.mode == 'AE':
            return - metrics.mean_absolute_error(pred, y)
        elif self.mode == '0-1':
            return metrics.accuracy_score(pred, y)
        else:
            raise NotImplementedError

if __name__ == '__main__':

    np.random.seed(1)
    from sklearn import datasets, metrics
    n_class = 5
    n_samples = 100


    X, y = datasets.make_regression(n_features=10, noise=80.5)
    bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
    y = np.digitize(y, bins[:-1])
    y -= np.min(y)

    X, y = datasets.make_classification(n_samples=n_samples,
        n_informative=20, n_classes=n_class, n_features=30)

    print X.shape
    print y
    print

    w, theta = threshold_fit(X, y, 0., n_class, mode='AE')
    pred = threshold_predict(X, w, theta)
    print metrics.mean_absolute_error(pred, y)
    print metrics.zero_one_loss(pred, y, normalize=True)
    print

    w, theta = threshold_fit(X, y, 0., n_class, mode='0-1')
    pred = threshold_predict(X, w, theta)
    print metrics.mean_absolute_error(pred, y)
    print metrics.zero_one_loss(pred, y, normalize=True)