"""
some ordinal regression algorithms
"""
import numpy as np
from scipy import optimize

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

def loss_logistic_immediate(x0, X, y, idx, y_unique):
    X = np.array(X)
    w = x0[:X.shape[1]]
    theta = x0[X.shape[1]:]
    Xw = X.dot(w)
    out = 0.
    for i in range(idx.shape[0]):

        if i == 0:
            f1 = theta[0] - Xw
            out += psi(f1)
        elif i == idx.shape[0] - 1:
            fk1 = theta[i-1] - Xw
            out += psi(-fk1)
        else:
            fy1 = theta[i-1] - Xw
            fy  = theta[i]   - Xw
            out += psi(-fy1) + psi(fy)
    return out


def logisitc_immediate(X, y, alpha):
    """
    Logisitc regression: immediate threshold
    """