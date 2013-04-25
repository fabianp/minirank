
from sklearn import datasets, linear_model, isotonic
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
    tmp = 0.
    for i in range(len(theta0) - 1):
        tmp += np.fmax(theta0[i] - theta0[i+1], 0) ** 2
    return - tmp

def f_grad(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])

    # gradient for w
    theta = theta0[y]
    a = elem_a(X, theta, w)
    b = elem_b(X, theta, w)
    tmp = (a * (1 - a) - b * (1 - b)) / (a - b)
    tmp = (X * tmp[:, None]).sum(0)

    # gradient for w2
    tmp_a = (a * (1 - a)) / (a - b)
    tmp_b = (b * (1 - b)) / (a - b)
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
    X = X[idx]
    y = y[idx].astype(np.int)
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y==u] = i
    x0 = np.ones(X.shape[1] + unique_y.size)
    x0[X.shape[1]:] = np.linspace(-1, 1, unique_y.size)

    check= optimize.check_grad(f_obj, f_grad, x0, X, y)
    assert check < 1.

    out = optimize.fmin_slsqp(f_obj, x0, args=(X, y), f_eqcons=f_eqcons,
                              f_ieqcons=f_ineqcons, fprime=f_grad, iter=10000)
    w, theta = np.split(out, [X.shape[1]])
    return w, theta[y]

def predict(w, theta, X):
    unique_theta = np.unique(theta)
    out = []
    for i in range(X.shape[0]):
        tmp = X[i].dot(w)
        tmp2 = np.abs(unique_theta - tmp)
        min_ = np.argsort(tmp2)[0]
        out.append(min_)
    return np.array(out)



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
    groups = y == y[:, None]
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    y_ = y.copy()
    for i in range(100):
        clf.fit(X, y_)
        z = clf.predict(X)
        y_ = isotonic_limited(z, groups, ymin=-1, ymax=1, x0=y_)
        y_ = np.array(y).ravel()
    return clf.coef_, y_

if __name__ == '__main__':
    X, y = load_data()
    w, theta = ordinal_logistic(X, y)
    pred = predict(w, theta, X)
    print 'Score %s' % (( pred == y).sum() / float(y.size))

    w_ls, theta_ls = ordinal_ls(X, y.astype(np.float), alpha=1e-3)
    pred_ls = predict(w_ls, theta_ls, X)
    #print 'Score LS %s' % (( pred_ls == y).sum() / float(y.size))
