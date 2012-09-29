import numpy as np
from scipy import stats
from ranking import train

def test_1():
    np.random.seed(0)
    X = np.random.randn(20, 5)
    w = np.random.randn(5)
    y = np.dot(X, w)
    w_ = train(X, y, 1., np.ones(20))

    tau, _ = stats.kendalltau(y, np.dot(X, w))
    assert np.abs(1 - tau) > 1e-3
