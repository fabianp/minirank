import numpy as np
from scipy import stats
from sklearn import base
from .sofia_ml import sgd_train

class RankSVM(base.BaseEstimator):

    def __init__(self, alpha=1., model='rank', max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, y, query_id=None):
        y = np.argsort(y)
        self.coef_, _ = sgd_train((X, y, query_id), self.alpha, max_iter=self.max_iter,
            model=self.model)
        return self

    def rank(self, X):
        order = np.argsort(np.dot(X, self.coef_))
        order_inv = np.zeros_like(order)
        order_inv[order] = np.arange(len(order))
        return order_inv

    # just so that GridSearchCV doesn't complain
    predict = rank

    def score(self, X, y):
        tau, _ = stats.kendalltau(np.dot(X, self.coef_), y)
        return np.abs(tau)
