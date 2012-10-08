import numpy as np
from scipy import stats
from minirank import train, sgd_predict

def test_1():
    np.random.seed(0)
    X = np.random.randn(200, 5)
    query_id = np.ones(len(X))
    w = np.random.randn(5)
    y = np.dot(X, w)
    coef, _ = train(X, y, 1., query_id, max_iter=100)
    prediction = sgd_predict(X, coef)
    tau, _ = stats.kendalltau(y, prediction)
    assert np.abs(1 - tau) > 1e-3

if __name__ == '__main__':
    test_1()