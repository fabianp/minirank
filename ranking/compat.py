import numpy as np
from scipy import stats
from sklearn import base
from ranking import train

class RankSVM(base.BaseEstimator):

    def __init__(self, alpha=1., max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, query_id=None):
        print 'training '
        self.coef_ = train(X, y, self.alpha, query_id, max_iter=self.max_iter)
        print 'done'
        return self

    def rank(self, X):
        return np.argsort(np.dot(X, self.coef_))

    def score(self, X, y):
        ord_est = self.rank(X)
        tau, _ = stats.kendalltau(ord_est, y)
        return tau