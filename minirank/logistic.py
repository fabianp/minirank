from __future__ import print_function

from sklearn import utils, metrics
from scipy import linalg, optimize, sparse
import numpy as np

BIG = 1e10

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


def ordinal_logistic(X, y, max_iter=1000, verbose=False):
    """
    Ordinal logistic regression or proportional odds model.
    Uses scipy's optimize.fmin_slsqp solver.

    Parameters
    ----------

    X : {array, sparse matrix}

    y : array-like

    max_iter : int
        maximum number of iterations
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


    x0 = np.ones(X.shape[1] + unique_y.size) / X.shape[0]
    x0[X.shape[1]:] = np.linspace(-1, 1, unique_y.size)

    if False:
        # check gradient is correctly computed
        check = optimize.check_grad(f_obj, f_grad, x0, X, y)
        print(optimize.approx_fprime(x0, f_obj, 1e-6, X, y))
        print(f_grad(x0, X, y))
        assert check < 1.

    bounds = [(-BIG, BIG)] * X.shape[1] + [(-1, 1)] * unique_y.size

    out = optimize.fmin_slsqp(f_obj, x0, args=(X, y),
                              f_ieqcons=f_ineqcons, fprime=f_grad, bounds=bounds,
                              fprime_ieqcons=f_ineqcons_grad, iter=max_iter, iprint=verbose)
    w, theta = np.split(out, [X.shape[1]])
    return w, theta[y][idx_inv]


def predict_logistic(w, theta, X):
    unique_theta = np.unique(theta)
    # import ipdb; ipdb.set_trace()
    mu = [-1]
    for i in range(unique_theta.size - 1):
        mu.append((unique_theta[i] + unique_theta[i+1]) / 2.)
        # todo: use roll

    out = X.dot(w)
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
    from sklearn import cross_validation
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

