"""
some ordinal regression algorithms
"""
import numpy as np
from scipy import optimize, linalg

def sigma(t):
    """
    logistic function, returns 1 / (1 + exp(-t))
    """
    return 1. / (1 + np.exp(-t))

def psi(t):
    """logistic loss"""
    return - np.log(sigma(t))

def psi_prime(t):
    """derivative of logistic loss"""
    return sigma(t) - 1

def loss_logistic_immediate(x0, X, idx, alpha):
    X = np.array(X)
    w = x0[:X.shape[1]]
    theta = x0[X.shape[1]:]
    Xw = X.dot(w)
    out = 0.
    for i in range(idx.shape[0]):
        Xwi = Xw[idx[i]]
        if i == 0:
            f1 = theta[0] - Xwi
            out += np.sum(psi(f1))
        elif i == idx.shape[0] - 1:
            fk1 = theta[i-1] - Xwi
            out += np.sum(psi(-fk1))
        else:
            fy1 = theta[i-1] - Xwi
            fy = theta[i] - Xwi
            out += np.sum(psi(-fy1) + psi(fy))
    return out + alpha * (linalg.norm(w) ** 2)


def grad_logistic_immediate(x0, X, idx):
    X = np.array(X)
    w = x0[:X.shape[1]]
    theta = x0[X.shape[1]:]
    Xw = X.dot(w)
    return

def logisitc_immediate(X, y, alpha):
    """
    Logisitc regression: immediate threshold
    """
    X = np.array(X)
    y = np.array(y)
    y_unique = np.unique(y)
    idx = (y_unique[:, None] == y)
    x0 = np.zeros(y_unique.shape[0] + X.shape[1])
    sol = optimize.minimize(loss_logistic_immediate, x0, args=(X, idx, alpha))
    print(sol)
    return sol.x


if __name__ == '__main__':
    n, p = 1000, 10
    X = 5 * np.random.randn(n, p)
    w = np.random.randn(p)
    y = np.floor(X.dot(w)).astype(np.int)
    y -= np.min(y)
    print y
    sol = logisitc_immediate(X, y, .1)
    print sol[:p] / w