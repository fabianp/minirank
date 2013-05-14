import numpy as np
from scipy import stats
from minirank import logistic

def test_logistic():
    n_samples, n_features = 10, 10
    X = np.random.randn(n_samples, n_features)
    y = np.arange(n_samples)
    w_, theta_ = logistic.ordinal_logistic_fit(X, y)
    pred = logistic.ordinal_logistic_predict(w_, theta_, X)
    assert np.all(pred == y)

if __name__ == '__main__':
    test_logistic()