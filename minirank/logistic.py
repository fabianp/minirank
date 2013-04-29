from __future__ import print_function

from sklearn import datasets, linear_model, metrics, cross_validation
from scipy import linalg, stats, optimize
import numpy as np
import pylab as pl

BIG = 1e10

def elem_a(X, theta, w):
    t = theta - X.dot(w)
    return t

def elem_b(X, theta, w):
    _theta = theta.copy()
    unique_theta = np.unique(theta)
    for i in range(len(unique_theta) - 1):
        t1 = unique_theta[i]
        t2 = unique_theta[i+1]
        _theta[_theta == t2] = t1
    _theta[theta == unique_theta[0]] = - np.inf
    t = _theta - X.dot(w)
    return t

def f_obj(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    theta = theta0[y]
    _theta = theta.copy()
    unique_theta = np.unique(theta)
    for i in range(len(unique_theta) - 1):
        t1 = unique_theta[i]
        t2 = unique_theta[i+1]
        _theta[_theta == t2] = t1
    _theta[theta == unique_theta[0]] = - np.inf

    a = elem_a(X, theta, w)
    b = elem_b(X, theta, w)
    tmp1 = np.dot(X, w) - np.log(np.exp(theta) - np.exp(_theta)) + \
           np.log((1 + np.exp(a))) + np.log((1 + np.exp(b)))
    #import ipdb; ipdb.set_trace()
    return tmp1.sum()


def f_ineqcons(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    return np.diff(theta0)

def f_ineqcons_grad(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    e1 = np.eye(theta0.size, 1).ravel()
    e2 = e1.copy()
    e2[1] = -1.
    L = - linalg.toeplitz(e1, e2)
    L = L[:-1]
    T = np.zeros((L.shape[0], x0.size))
    T[:, x0.size - L.shape[1]:] = L[:, :]
    return T

def f_grad(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])

    # gradient for w
    theta = theta0[y]
    a = elem_a(X, theta, w)
    a = 1. / (1 + np.exp(-a))
    b = elem_b(X, theta, w)
    b = 1. / (1 + np.exp(-b))
    quot = (a - b)
    quot[quot == 0] = 1e-32

    # gradient for w2
    tmp_a = (a * (1 - a)) / quot
    tmp_b = (b * (1 - b)) / quot
    tmp3 = np.zeros(theta0.size)
    for i in range(y.size):
        e = np.zeros(theta0.size)
        e1 = e.copy()
        e[y[i]] = 1.
        tmp3 += tmp_a[i] * e
        if y[i] > 0:
            e1[y[i] - 1] = 1.
            tmp3 -= tmp_b[i] * e1

    tmp1 = ((1 - a - b) * X.T).sum(1)
    out = np.concatenate((tmp1, - tmp3))
    #import ipdb; ipdb.set_trace()
    return out

def ordinal_logistic(X, y, max_iter=1000):
    idx = np.argsort(y)
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = np.arange(idx.size)

    X = X[idx]
    y = y[idx].astype(np.int)
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y == u] = i
    x0 = np.ones(X.shape[1] + unique_y.size)
    x0[X.shape[1]:] = np.linspace(-1, 1, unique_y.size)

    check = optimize.check_grad(f_obj, f_grad, x0, X, y)
    assert check < 1.

    bounds = [(-BIG, BIG)] * X.shape[1] + [(-1, 1)] * unique_y.size

    out = optimize.fmin_slsqp(f_obj, x0, args=(X, y),
                              f_ieqcons=f_ineqcons, fprime=f_grad, bounds=bounds,
                              fprime_ieqcons=f_ineqcons_grad, iter=max_iter)
    w, theta = np.split(out, [X.shape[1]])
    return w, theta[y][idx_inv]

def predict_logistic(w, theta, X):
    unique_theta = np.unique(theta)
    # import ipdb; ipdb.set_trace()
    mu = [-1]
    for i in range(unique_theta.size - 1):
        mu.append((unique_theta[i] + unique_theta[i+1]) / 2.)
        # todo: use roll

    out = np.dot(X, w)
    mu = np.array(mu)
    tmp = metrics.pairwise.pairwise_distances(out[:, None], mu[:, None])
    return np.argmin(tmp, 1)

def load_data():
    data = np.loadtxt('minirank/data/pyrim.ord')
    X = data[:, :-1]
    X -= X.mean()
    y = data[:, -1]
    y -= y.min()
    return X, y

if __name__ == '__main__':
    X, y = load_data()
    idx = np.argsort(y)
    X = X[idx]
    y = y[idx]
    cv = cross_validation.StratifiedShuffleSplit(y, n_iter=50, test_size=.25, random_state=0)
    score_logistic = []
    score_ordinal_logistic = []
    for i, (train, test) in enumerate(cv):
        assert np.all(np.unique(y[train]) == np.unique(y))
        train = np.sort(train)
        test = np.sort(test)
        w, theta = ordinal_logistic(X[train], y[train])
        pred = predict_logistic(w, theta, X[test])
        s = ((pred == y[test]).sum() / float(test.size))
        print('Score (ORDINAL)  fold %s: %s' % (i+1, s))
        score_ordinal_logistic.append(s)

        from sklearn import linear_model
        clf = linear_model.LogisticRegression(C=1e10)
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        s = ((pred == y[test]).sum() / float(test.size))
        print('Score (LOGISTIC) fold %s: %s' % (i+1, s))
        score_logistic.append(s)

    print()
    print('MEAN SCORE (ORDINAL LOGISTIC):    %s' % np.mean(score_ordinal_logistic))
    print('MEAN SCORE (LOGISTIC REGRESSION): %s' % np.mean(score_logistic))
    print('Chance level is at %s' % (1. / np.unique(y).size))

