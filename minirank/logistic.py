"""
Implementation of logistic ordinal regression (aka proportional odds) model
"""

from __future__ import print_function

from sklearn import utils, metrics
from scipy import linalg, optimize, sparse
import numpy as np

BIG = 1e10


def ordinal_logistic_fit(X, y, max_iter=1000, verbose=False):
    """
    Ordinal logistic regression or proportional odds model.
    Uses scipy's optimize.fmin_slsqp solver.

    Parameters
    ----------
    X : {array, sparse matrix}, shape (n_samples, n_feaures)
        Input data
    y : array-like
        Target values
    max_iter : int
        Maximum number of iterations
    verbose: bool
        Print convergence information

    Returns
    -------
    w : array, shape (n_features,)
        coefficients of the linear model
    theta : array, shape (k,), where k is the different values of y
        vector of thresholds
    """

    X = utils.safe_asarray(X)
    y = np.asarray(y)

    # .. order input ..
    idx = np.argsort(y)
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = np.arange(idx.size)
    X = X[idx]
    y = y[idx].astype(np.int)
    # make them continuous and start at zero
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y == u] = i
    unique_y = np.unique(y)

    # .. utility arrays used in f_grad ..
    E0 = (y[:, np.newaxis] == np.unique(y)).astype(np.int)
    E1 = np.roll(E0, -1, axis=-1)
    E1[:, -1] = 0.
    E0, E1 = map(sparse.csr_matrix, (E0.T, E1.T))
    L = np.tril(np.ones((unique_y.size, unique_y.size)))

    def f_obj(x0, X, y):
        """
        Objective function
        """
        w, z = np.split(x0, [X.shape[1]])
        theta0 = L.dot(z)
        theta1 = np.roll(theta0, 1)  # theta_{y_i - 1}
        theta1[0] = - np.inf

        Xw = X.dot(w)
        a = theta0[y] - Xw
        b = theta1[y] - Xw
        out = np.zeros(a.size, dtype=np.float)
        idx = a < 0
        a0, b0 = a[idx], b[idx]
        out[idx] = a0 + np.log(1 - np.exp(b0 - a0)) - \
                    np.log(1 + np.exp(a0)) - np.log(1 + np.exp(b0))
        a0, b0 = a[~idx], b[~idx]
        out[~idx] = a0 + np.log(1 - np.exp(b0 - a0)) + \
            1 - a0 - np.log(1 + np.exp(-a0)) + 1 - b0 - np.log(1 + np.exp(-b0))

        a0 = a[y == 0]
        if a0[0] > 0:
            out[y == 0] = 1 - np.log(1 + np.exp(- a0))
        else:
            out[y == 0] = a0 - np.log(1 + np.exp(a0))
#        print((~ np.isfinite(out)).sum())
        if (~np.isfinite(out)).sum() > 0:
            pass
            #import ipdb; ipdb.set_trace()
        return - out.sum()

    def f_grad(x0, X, y):
        """
        Gradient of the objective function
        """
        w, z = np.split(x0, [X.shape[1]])
        theta0 = L.dot(z)
        theta1 = np.roll(theta0, 1)  # theta_{y_i - 1}
        theta1[0] = - np.inf
        Xw = X.dot(w)
        a = Xw - theta0[y]
        b = Xw - theta1[y]

        # gradient for w
        a = 1. / (1 + np.exp(a))
        b = 1. / (1 + np.exp(b))
        grad_w = np.sum((1 - a - b) * X.T, axis=1)

        tmp0 = 1 - a - \
            1. / (1 - np.exp(theta0[y] - theta1[y]))
        tmp0 = E0.dot(tmp0)
        tmp1 = 1 - b - \
            1. / (1 - np.exp(- (theta0[y] - theta1[y])))
        tmp1 = E1.dot(tmp1)
        grad_theta = - (tmp0 + tmp1)
#        import ipdb; ipdb.set_trace()

        out = np.concatenate((grad_w, L.T.dot(grad_theta)))
        return out


    x0 = np.random.randn(X.shape[1] + unique_y.size) / X.shape[1]
    x0 = np.zeros(X.shape[1] + unique_y.size) / X.shape[1]
    x0[X.shape[1]] = -1.
    x0[X.shape[1] + 1:] = 1. / unique_y.size

    if True:
        # check that gradient is correctly computed
        check = optimize.check_grad(f_obj, f_grad, x0, X, y)
        approx_grad = optimize.approx_fprime(x0, f_obj, 1e-3, X, y)
        #print(approx_grad[X.shape[1]:])
        #print(f_grad(x0, X, y)[X.shape[1]:])
        #print(check, f_obj(x0, X, y))
        assert np.abs(check / f_obj(x0, X, y)) < 1e-3

    bounds = [(None, None)] * (X.shape[1] + 1) + [(1. / unique_y.size, None)] * (unique_y.size - 1)

    out = optimize.fmin_tnc(f_obj, x0, args=(X, y),
            fprime=f_grad, bounds=bounds, maxfun=max_iter, disp=0)
    w, z = np.split(out[0], [X.shape[1]])
    theta = L.dot(z)
    #import ipdb; ipdb.set_trace()
    return w, theta[y][idx_inv]

def ordinal_logistic_predict(w, theta, X, y):
    """
    Parameters
    ----------
    w : coefficients obtained by ordinal_logistic
    theta : thresholds
    """
    unique_y = np.sort(np.unique(y))
    unique_theta = np.unique(theta)
    mu = [-1]
    for i in range(unique_theta.size - 1):
        mu.append((unique_theta[i] + unique_theta[i+1]) / 2.)
        # todo: use roll
    out = X.dot(w)
    mu = np.array(mu)
    tmp = metrics.pairwise.pairwise_distances(out[:, None], mu[:, None])
    return unique_y[np.argmin(tmp, 1)]



def ordinal_logistic_bound_fit(X, y, max_iter=1000, verbose=False):
    """
    Ordinal logistic regression or proportional odds model.
    Uses scipy's optimize.fmin_slsqp solver.

    Parameters
    ----------
    X : {array, sparse matrix}, shape (n_samples, n_feaures)
        Input data
    y : array-like
        Target values
    max_iter : int
        Maximum number of iterations
    verbose: bool
        Print convergence information

    Returns
    -------
    w : array, shape (n_features,)
        coefficients of the linear model
    theta : array, shape (k,), where k is the different values of y
        vector of thresholds
    """

    X = utils.safe_asarray(X)
    y = np.asarray(y)

    # .. order input ..
    idx = np.argsort(y)
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = np.arange(idx.size)
    X = X[idx]
    y = y[idx].astype(np.int)
    # make them continuous and start at zero
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y == u] = i
    unique_y = np.unique(y)

    # .. utility arrays used in f_grad ..
    E0 = (y[:, np.newaxis] == np.unique(y)).astype(np.int)
    E1 = np.roll(E0, -1, axis=-1)
    E1[:, -1] = 0.
    E0, E1 = map(sparse.csr_matrix, (E0.T, E1.T))

    def f_obj(x0, X, y):
        """
        Objective function
        """
        w, theta0 = np.split(x0, [X.shape[1]])
        theta1 = np.roll(theta0, 1)  # theta_{y_i - 1}
        theta1[0] = - np.inf

        Xw = X.dot(w)
        a = Xw - theta0[y]
        b = Xw - theta1[y]
        tmp = np.log(1. / (1 + np.exp(a)) - 1. / (1 + np.exp(b)))
        return - tmp.sum()

    def f_grad(x0, X, y):
        """
        Gradient of the objective function
        """
        w, theta0 = np.split(x0, [X.shape[1]])
        theta1 = np.roll(theta0, 1)  # theta_{y_i - 1}
        theta1[0] = - np.inf
        Xw = X.dot(w)
        a = Xw - theta0[y]
        b = Xw - theta1[y]

        # gradient for w
        a = 1. / (1 + np.exp(a))
        b = 1. / (1 + np.exp(b))
        grad_w = np.sum((1 - a - b) * X.T, axis=1)

        tmp0 = 1 - 1. / (1 + np.exp(a)) - \
               1. / (1 - np.exp(theta1[y] - theta0[y]))
        tmp0 = E0.dot(tmp0)
        tmp1 = 1 - 1. / (1 + np.exp(b)) - \
               1. / (1 - np.exp(- (theta1[y] - theta0[y])))
        tmp1 = E1.dot(tmp1)
        grad_theta = tmp0 + tmp1

        return np.concatenate((grad_w, grad_theta))

    def f_ineqcons(x0, X, y):
        w, theta0 = np.split(x0, [X.shape[1]])
        out = np.diff(theta0)
        return out

    def f_ineqcons_grad(x0, X, y):
        w, theta0 = np.split(x0, [X.shape[1]])
        e1 = np.eye(theta0.size, 1).ravel()
        e2 = e1.copy()
        e2[1] = -1.
        L = - linalg.toeplitz(e1, e2)
        L = L[:-1]
        T = np.zeros((L.shape[0], x0.size))
        T[:, x0.size - L.shape[1]:] = L[:, :]
        return -T

    x0 = np.ones(X.shape[1] + unique_y.size) / X.shape[0]
    x0[X.shape[1]:] = np.linspace(-.9, .9, unique_y.size)

    if False:
        # check that gradient is correctly computed
        check = optimize.check_grad(f_obj, f_grad, x0, X, y)
        assert check / f_obj(x0, X, y) < 1.

    bounds = [(-BIG, BIG)] * X.shape[1] + [(-1, 1)] * unique_y.size

    out = optimize.fmin_slsqp(f_obj, x0, args=(X, y),
                              f_ieqcons=f_ineqcons, fprime=f_grad, bounds=bounds,
                              fprime_ieqcons=f_ineqcons_grad, iter=max_iter, iprint=verbose)
    w, theta = np.split(out, [X.shape[1]])
    return w, theta[y][idx_inv]

if __name__ == '__main__':
    DOC = """
================================================================================
    Compare the prediction accuracy of different models on the boston dataset
================================================================================
    """
    print(DOC)
    from sklearn import cross_validation, datasets
    boston = datasets.load_boston()
    X, y = boston.data, np.round(boston.target)
    #X -= X.mean()
    y -= y.min()

    idx = np.argsort(y)
    X = X[idx]
    y = y[idx]
    cv = cross_validation.ShuffleSplit(y.size, n_iter=50, test_size=.1, random_state=0)
    score_logistic = []
    score_ordinal_logistic = []
    score_ridge = []
    for i, (train, test) in enumerate(cv):
        #test = train
        if not np.all(np.unique(y[train]) == np.unique(y)):
            # we need the train set to have all different classes
            continue
        assert np.all(np.unique(y[train]) == np.unique(y))
        train = np.sort(train)
        test = np.sort(test)
        w, theta = ordinal_logistic_fit(X[train], y[train])
        pred = ordinal_logistic_predict(w, theta, X[test], y)
        s = metrics.mean_absolute_error(y[test], pred)
        print('ERROR (ORDINAL)  fold %s: %s' % (i+1, s))
        score_ordinal_logistic.append(s)

        from sklearn import linear_model
        clf = linear_model.LogisticRegression(C=1.)
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        s = metrics.mean_absolute_error(y[test], pred)
        print('ERROR (LOGISTIC) fold %s: %s' % (i+1, s))
        score_logistic.append(s)

        from sklearn import linear_model
        clf = linear_model.Ridge(alpha=1.)
        clf.fit(X[train], y[train])
        pred = np.round(clf.predict(X[test]))
        s = metrics.mean_absolute_error(y[test], pred)
        print('ERROR (RIDGE)    fold %s: %s' % (i+1, s))
        score_ridge.append(s)


    print()
    print('MEAN ABSOLUTE ERROR (ORDINAL LOGISTIC):    %s' % np.mean(score_ordinal_logistic))
    print('MEAN ABSOLUTE ERROR (LOGISTIC REGRESSION): %s' % np.mean(score_logistic))
    print('MEAN ABSOLUTE ERROR (RIDGE REGRESSION):    %s' % np.mean(score_ridge))
    # print('Chance level is at %s' % (1. / np.unique(y).size))