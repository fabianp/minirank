
from sklearn import datasets, linear_model, isotonic
from scipy import linalg, stats, optimize
import numpy as np
import cvxopt
import pylab as pl

BIG = 1e10

def load_data():
    data = np.loadtxt('minirank/data/pyrim.ord')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def elem_a(X, theta, w):
    t = theta - X.dot(w)
    return 1. / (1 + np.exp(t))

def elem_b(X, theta, w):
    _theta = np.roll(theta, 1)
    _theta[0] = - np.inf
    t = theta - X.dot(w)
    return 1. / (1 + np.exp(t))

def f_obj(x0, X, y):
    w, theta0 = np.split(x0, [X.shape[1]])
    theta = theta0[y]
    tmp2 = np.sum(elem_a(X, theta, w) - elem_b(X, theta, w))
    tmp2 += BIG * ((theta[0] + 1) ** 2)
    tmp2 += BIG * ((theta[-1] - 1) ** 2)
    for i in range(theta0.size - 1):
        tmp2 += BIG * (np.fmax(theta[i] - theta[i+1], 0) ** 2)
    return tmp2

def ordinal_logistic(X, y):
    idx = np.argsort(y)
    X = X[idx]
    y = y[idx].astype(np.int)
    unique_y = np.unique(y)
    for i, u in enumerate(unique_y):
        y[y==u] = i
    x0 = np.zeros(X.shape[1] + unique_y.size)
    out = optimize.minimize(f_obj, x0, args=(X, y))
    w, theta = np.split(out.x, [X.shape[1]])
    return w, theta[y]


if __name__ == '__main__':
    X, y = load_data()
    w, theta = ordinal_logistic(X, y)
    print w
    print theta