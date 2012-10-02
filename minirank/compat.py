import numpy as np
from scipy import stats
from sklearn import base
from .sofia_ml import train

class RankSVM(base.BaseEstimator):

    def __init__(self, alpha=1., model='rank', max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, y, query_id=None):
        y = np.argsort(y)
        self.coef_ = train(X, y, self.alpha, query_id, max_iter=self.max_iter,
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


#### JUST FOR DEBUGGING


import itertools

from sklearn import svm, linear_model


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    diff: array
        target difference for each considered samples
    """
    X_new = []
    y_new = []
    diff = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    for (i, j) in comb:
        #if np.abs(y[i, 0] - y[j, 0]) <= 1. or y[i, 1] != y[j, 1]:
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        diff.append(y[i, 0] - y[j, 0])
        y_new.append(np.sign(diff[-1]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
            diff[-1] = - diff[-1]
        k += 1
    return np.asarray(X_new), np.asarray(y_new).ravel(), np.array(diff).ravel()


class RankSVM2(svm.SVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.SVC` for a full description of parameters.
    """
    def __init__(self, C=1.0, degree=3,
                 shrinking=True, probability=False,
                 tol=1e-3, cache_size=200):

        super(RankSVM2, self).__init__(kernel='linear', degree=degree,
            tol=tol, C=C, shrinking=shrinking, probability=probability,
            cache_size=cache_size)

    def fit(self, X, y, sample_weight=False):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        sample_weights : boolean
           wether to consider sample weights in the ranking problem
           (= weighted loss function)

        Returns
        -------
        self
        """
        X_trans, y_trans, diff = transform_pairwise(X, y)
        if sample_weight:
            super(RankSVM2, self).fit(X_trans, y_trans, sample_weight=20 * np.abs(diff))
        else:
            super(RankSVM2, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.dot(X, self.coef_.T).ravel()
        else:
            raise ValueError("Must call fit() prior to rank()")

    def score(self, X, y):
        """
        Because we transformed into a balanced pairwise problem, chance level is at 0.5
        """
        tau, _ = stats.kendalltau(self.predict(X), y)
        return tau