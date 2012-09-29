import numpy as np
from scipy import stats
from sklearn import base
from ranking import train

class RankSVM(base.BaseEstimator):

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y, query_id=None):
        self.coef_ = train(X, y, self.alpha, query_id)

    def rank(self, X):
        return np.argsort(np.dot(X, self.coef_))

    def score(self, X, y):
        ord_est = self.rank(X)
        tau, _ = stats.kendalltau(ord_est, y)
        return tau