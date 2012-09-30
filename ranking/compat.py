import numpy as np
from scipy import stats
from sklearn import base
from ranking import train

class RankSVM(base.BaseEstimator):

    def __init__(self, alpha=1., max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, query_id=None):
        self.coef_ = train(X, y, self.alpha, query_id, max_iter=self.max_iter)
        return self

    def rank(self, X):
        return np.argsort(np.dot(X, self.coef_))

    # just so that GridSearchCV doesn't complain
    predict = rank

    def score(self, X, y):
        tau, _ = stats.kendalltau(np.dot(X, self.coef_), y)
        return tau