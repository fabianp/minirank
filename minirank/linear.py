"""
Adapt linear regression models to the setting of ordinal regression.

In practice this means override the predict method so that it predicts
ordinal (not continuous) labels. Also the score method is overwritten to
use the mean absolute error.
"""

from sklearn import linear_model, metrics, svm
import numpy as np
import metrics as m  # avoid name clash with sklearn.metrics


METRIC = lambda x, y: - metrics.mean_absolute_error(x, y)
# METRIC = lambda x, y: - metrics.zero_one_loss(x, y, normalize=True)
# METRIC = lambda x, y: - m.pairwise_disagreement(x, y)

class RidgeOR(linear_model.Ridge):
    """
    Overwrite Ridge from scikit-learn to use
    the (minus) absolute error as score function.

    (see https://github.com/scikit-learn/scikit-learn/issues/3848
    on why this cannot be accomplished using a GridSearchCV object)
    """

    def fit(self, X, y):
        self.unique_y_ = np.unique(y)
        super(linear_model.Ridge, self).fit(X, y)
        return self

    def predict(self, X):
        pred =  np.round(super(linear_model.Ridge, self).predict(X))
        pred = np.clip(pred, 0, self.unique_y_.max())
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return METRIC(y, pred)



class LAD(svm.LinearSVR):
    """
    Least Absolute Deviation
    """

    def fit(self, X, y):
        self.unique_y_ = np.unique(y)
        svm.LinearSVR.fit(self, X, y)
        return self

    def predict(self, X):
        pred = np.round(super(svm.LinearSVR, self).predict(X))
        pred = np.clip(pred, 0, self.unique_y_.max())
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return METRIC(y, pred)


