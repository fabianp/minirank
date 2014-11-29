"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize, linalg, stats

from sklearn import base, metrics

from metrics import pairwise_disagreement

from joblib import Memory


METRIC = lambda x, y: - metrics.mean_absolute_error(x, y)
# METRIC = lambda x, y: - metrics.zero_one_loss(x, y, normalize=True)
# METRIC = lambda x, y: - pairwise_disagreement(x, y)

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
    #theta = np.sort(theta)
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
    #theta = np.sort(theta)
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



def obj_multiclass(x0, X, y, alpha, n_class):
    n_samples, n_features = X.shape
    W = x0.reshape((n_features + 1, n_class))
    


def threshold_fit(X, y, alpha, n_class, mode='AE', verbose=False, 
                  maxiter=10000, bounds=False):
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
    x0[X.shape[1]:] = np.arange(n_class - 1)
    if bounds == True:
        bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_class - 2)
    else:
        bounds = None
    options = {'maxiter' : maxiter}
    sol = optimize.minimize(obj_margin, x0, jac=grad_margin,
        args=(X, y, alpha, n_class, weights), method='L-BFGS-B', bounds=bounds,
        options=options)
    if not sol.success:
        print(sol.message)
    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = np.cumsum(c)
    return w, np.sort(theta)


def threshold_predict(X, w, theta):
    """
    Class numbers are between 0 and k-1
    """
    idx = np.concatenate((np.argsort(theta), [theta.size]))
    pred = []
    n_samples = X.shape[0]
    Xw = X.dot(w)
    tmp = theta[:, None] - Xw
    pred = np.sum(tmp <= 0, axis=0).astype(np.int)
    return pred


def multiclass_fit(X, y, alpha, n_class):


class MarginOR(base.BaseEstimator):
    def __init__(self, n_class=2, alpha=1., mode='AE', scoring='AE', 
        verbose=0, maxiter=10000):
        self.alpha = alpha
        self.mode = mode
        self.scoring = scoring
        self.n_class = n_class
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        self.w_, self.theta_ = threshold_fit(X, y, self.alpha, self.n_class,
            mode=self.mode, verbose=self.verbose)
        return self

    def predict(self, X):
        return threshold_predict(X, self.w_, self.theta_)

    def score(self, X, y):
        pred = self.predict(X)
        return METRIC(pred, y)


if __name__ == '__main__':

    np.random.seed(0)
    from sklearn import datasets, metrics, svm, cross_validation
    n_class = 5
    n_samples = 100


    X, y = datasets.make_classification(n_samples=n_samples,
        n_informative=20, n_classes=n_class, n_features=100)

    print X.shape
    print y
    print

    #w, theta = threshold_fit(X, y, 0., n_class, mode='AE')
    #pred = threshold_predict(X, w, theta)
    #print metrics.mean_absolute_error(pred, y)
    #print metrics.zero_one_loss(pred, y, normalize=True)
    #print

    cv = cross_validation.KFold(y.size)
    for train, test in cv:
        w, theta = threshold_fit(X[train], y[train], 1., n_class, mode='0-1', 
                                 bounds=False)
        print(np.argsort(theta))
        print(theta)
        pred = threshold_predict(X[test], w, theta)
        # print metrics.mean_absolute_error(pred, y)
        print metrics.accuracy_score(pred, y[test])
        
        pred = svm.LinearSVC().fit(X[train], y[train]).predict(X[test])
        print metrics.accuracy_score(pred, y[test])
        print
