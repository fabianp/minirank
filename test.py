import numpy as np
from scipy import stats
from minirank import train

def test_1():
    np.random.seed(0)
    X = np.random.randn(200, 5)
    query_id = np.ones(len(X))
    w = np.random.randn(5)
    y = np.dot(X, w)
    w_ = train(X, y, 1., query_id, max_iter=100)
    tau, _ = stats.kendalltau(y, np.dot(X, w_))
    assert np.abs(1 - tau) > 1e-3

if __name__ == '__main__':
    test_1()