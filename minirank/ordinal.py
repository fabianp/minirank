
from sklearn import datasets, linear_model, metrics, cross_validation
from scipy import linalg, stats, optimize
import numpy as np
import pylab as pl

BIG = 1e10

def load_data():
    data = np.loadtxt('minirank/data/pyrim.ord')
    X = data[:, :-1]
    y = data[:, -1]
    y -= y.min()
    return X, y.astype(np.int)

def elem_a(X, theta, w):
    t = theta - X.dot(w)
    return 1. / (1 + np.exp(-t))

def elem_b(X, theta, w):
    _theta = theta.copy()
    unique_theta = np.unique(theta)
    for i in range(len(unique_theta) - 1):
        t1 = unique_theta[i]
        t2 = unique_theta[i+1]
        _theta[_theta == t2] = t1
    _theta[theta == unique_theta[0]] = - np.inf
    t = _theta - X.dot(w)
    return 1. / (1 + np.exp(-t))

def f_obj(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    theta = theta0[y]
    tmp = elem_a(X, theta, w) - elem_b(X, theta, w)
    tmp2 = - np.sum(np.log(tmp))
    return tmp2

def f_eqcons(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    return (theta0[0] + 1.) ** 2 + (theta0[-1] - 1.) ** 2

def f_ineqcons(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    # print theta0
    return np.diff(theta0).min()

def f_grad(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])

    # gradient for w
    theta = theta0[y]
    a = elem_a(X, theta, w)
    b = elem_b(X, theta, w)
    quot = (a - b)
    quot[quot == 0] = 1e-32
    tmp = (a * (1 - a) - b * (1 - b)) / quot
    tmp = (X * tmp[:, None]).sum(0)
#    if np.any((a-b) == 0):
#        pass
#        import ipdb; ipdb.set_trace()

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
    #import ipdb; ipdb.set_trace()
    return np.concatenate((tmp, - tmp3))

def ordinal_logistic(X, y):
    idx = np.argsort(y)
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = np.arange(len(idx))

    X = X[idx]
    y = y[idx].astype(np.int)
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y == u] = i
    x0 = np.ones(X.shape[1] + unique_y.size)
    x0[X.shape[1]:] = np.linspace(-1, 1, unique_y.size)

    check= optimize.check_grad(f_obj, f_grad, x0, X, y)
    assert check < 1.

    out = optimize.fmin_slsqp(f_obj, x0, args=(X, y), f_eqcons=f_eqcons,
                              f_ieqcons=f_ineqcons, fprime=f_grad, iter=5000)
    w, theta = np.split(out, [X.shape[1]])
    return w, theta[y][idx_inv]

def predict_logistic(w, theta, X):
    unique_theta = np.unique(theta)
    # import ipdb; ipdb.set_trace()
    mu = [-1]
    for i in range(unique_theta.size - 1):
        mu.append((unique_theta[i] + unique_theta[i+1]) / 2.)

    out = np.dot(X, w)
    mu = np.array(mu)
    tmp = metrics.pairwise.pairwise_distances(out[:, None], mu[:, None])
    return np.argmin(tmp, 1)

def predict_ls(w, mu, X):
    out = np.dot(X, w)
    unique_mu = np.unique(mu)
    # get the closest one in theta
    tmp = metrics.pairwise.pairwise_distances(out[:, None], unique_mu[:, None])
    return np.argmin(tmp, 1)


def isotonic_limited(y, groups, ymin=-1, ymax=1, x0=None, verbose=False):
    import cvxopt
    import cvxopt.solvers
    cvxopt.solvers.options['show_progress'] = verbose
    A = np.zeros((y.size, y.size))
    for i in range(groups.shape[0]):
        idx = np.where(groups[i])[0]
        if len(idx) < 2:
            continue
        tmp1 = [1, -1] + [0] * (len(idx) - 2)
        tmp2 = [1,  0] + [0] * (len(idx) - 2)
        A[idx[:, None], idx] = linalg.toeplitz(tmp2, tmp1)
        A[idx[-1], idx[-1]] = 0.
    A = A[(A * A).sum(1) > 0]
    G = linalg.toeplitz([1.] + [0] * (y.size - 1), [1, -1] + [0] * (y.size - 2))
    G = np.concatenate((np.eye(1, G.shape[1]), G), axis=0)
    h = np.zeros(y.size + 1)
    h[0] = -1.
    h[-1] = 1.
    A_cvx = cvxopt.matrix(A)
    P_cvx = cvxopt.matrix(np.eye(y.size))
    q_cvx = cvxopt.matrix(- y.ravel())
    b_cvx = cvxopt.matrix(np.zeros_like(A[:, 0]))
    G_cvx = cvxopt.matrix(G)
    h_cvx = cvxopt.matrix(h)
    initvals = {}
    if x0 is not None:
        initvals['x'] = cvxopt.matrix(x0)
    sol = cvxopt.solvers.qp(P_cvx, q_cvx, A=A_cvx, b=b_cvx, G=G_cvx, h=h_cvx, initvals=initvals)
    return sol['x']


def ordinal_ls(X, y, alpha=0.):
    y = np.array(y, dtype=np.float)
    idx = np.argsort(y)
    idx_inv = np.zeros_like(idx)
    idx_inv[idx] = np.arange(len(idx))
    X = X[idx]
    y = y[idx]
    groups = y == y[:, None]
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    y_ = y.copy()
    for i in range(100):
        clf.fit(X, y_)
        z = clf.predict(X)
        y_ = isotonic_limited(z, groups, ymin=-1, ymax=1, x0=y_)
        y_ = np.array(y).ravel()
    return clf.coef_, y_[idx_inv]

if __name__ == '__main__':
    X, y = load_data()
    cv = cross_validation.StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
    score_logistic = []
    score_ls = []
    for train, test in cv:
        w, theta = ordinal_logistic(X[train], y[train])
        pred = predict_logistic(w, theta, X[test])
        s = ((pred == y[test]).sum() / float(test.size))
        print s
        #print pred
        #print y[test]
        score_logistic.append(s)


        w_ls, mu_ls = ordinal_ls(X[train], y[train].astype(np.float), alpha=1e-3)
        pred_ls = predict_ls(w_ls, mu_ls, X[test])
        #print pred_ls
        s = ((pred_ls == y[test]).sum() / float(test.size))
        score_ls.append(s)
        print s
        print

    print 'Score LOGISTIC %s' % np.mean(score_logistic)
    print 'Score LS       %s' % np.mean(score_ls)
