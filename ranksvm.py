"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
"""

import itertools
import numpy as np
from scipy import linalg

from sklearn import svm, linear_model, cross_validation


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


class RankSVM(svm.SVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.SVC` for a full description of parameters.
    """
    def __init__(self, C=1.0, degree=3,
                 shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, scale_C=True):

        super(RankSVM, self).__init__(kernel='linear', degree=degree,
            tol=tol, C=C, shrinking=shrinking, probability=probability,
            cache_size=cache_size, scale_C=scale_C)

    def fit(self, X, y, sample_weight=True):
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
            super(RankSVM, self).fit(X_trans, y_trans, sample_weight=20 * np.abs(diff))
        else:
            super(RankSVM, self).fit(X_trans, y_trans)
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
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a balanced pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans, diff = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)

class RankLogistic(linear_model.LogisticRegression):

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
            super(RankLogistic, self).fit(X_trans, y_trans, sample_weight=diff)
        else:
            super(RankLogistic, self).fit(X_trans, y_trans)
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
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a balanced pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans, diff = transform_pairwise(X, y)
        return np.mean(super(RankLogistic, self).predict(X_trans) == y_trans)


def linear_rank(X, y, alphas):
    """
    Fit athe "value-regularized" linear method described in the paper
    "On the consistency of Ranking Algorithms", Duchi et al.

    Parameters
    ----------
    alpha: l2-term regularization
    """

    alphas = np.array(alphas)
    U, s, Vt = linalg.svd(X, full_matrices=False)
    Xp, yp, d = transform_pairwise(X, y)
    #d = np.sign(d) * ((100 * d) ** 2)
    #omega = 1. / ( 1e3 * X.shape[0] ** 2)
    #omega = 1e-4
    v =  np.dot(Vt, np.dot(d, Xp))
    return np.dot(v / ((s ** 2) + alphas[:, np.newaxis]), Vt)


def test_linear_rank():
    X = np.random.randn(5, 5)
    y = np.random.randn(5)
    alphas = [.1, 1., 2.]
    W = linear_rank(X, y, alphas)
    Xp, yp, d = transform_pairwise(X, y)

    # check KKT conditions
    for i in range(3):
        right = np.dot(np.dot(X.T, X) + alphas[i] * np.eye(5), W[i])
        left = np.dot(Xp.T, d)
        assert np.allclose(right, left)
    print 'TESTS OK'



class LinearRankCV(linear_model.base.LinearModel):
    """
    value-regularized linear method described in the paper
    "On the consistency of Ranking Algorithms", Duchi et al.
    """
    def __init__(self, alphas):
        self.alphas = alphas
        self.fit_intercept = True

    def fit(self, X, y):
        X, y, X_mean, y_mean, X_std =\
        self._center_data(X, y, self.fit_intercept,
            False, True)
        cv = cross_validation.KFold(X.shape[0], 5)
        scores = np.zeros(len(self.alphas))
        for train, test in cv:
            Xp, yp, dp = transform_pairwise(X[train], y[train])
            W = linear_rank(X[train], y[train], self.alphas)
            Xt, yt, dt = transform_pairwise(X[test], y[test])
            if not yt.size:
                continue

            train_score = dp * np.dot(Xp, W.T).T
            assert train_score.shape == (len(self.alphas), Xp.shape[0])
            #print(np.mean(dp * np.dot(Xp, W.T).T > 0, 1))

            scores += np.mean(dt * np.dot(Xt, W.T).T > 0, 1)

        self.best_alpha = self.alphas[np.argmax(scores)]
        self.coef_ = linear_rank(X, y, [self.best_alpha])[0] # learn on whole dataset
        self._set_intercept(X_mean, y_mean, X_std)
        return self

    def score(self, X, y):
        Xt, yt, dt = transform_pairwise(X, y)
        return np.mean(np.sign(np.dot(Xt, self.coef_.T)) == yt)

    def transform(self, X):
        return np.dot(X, self.coef_)[:, np.newaxis]



if __name__ == '__main__':
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    np.random.seed(0)
    n_samples, n_features = 300, 5
    true_coef = np.random.randn(n_features)
    X = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
    y = np.dot(X, true_coef)
    y = np.sqrt(y - np.min(y))  # add non-linearities
    y += .1 * noise  # add noise
    Y = np.c_[y, np.mod(np.arange(n_samples), 5)]  # add query fake id
    cv = cross_validation.KFold(n_samples, 5)
    train, test = iter(cv).next()

    # make a simple plot out of it
    import pylab as pl
    pl.scatter(np.dot(X, true_coef), y)
    pl.title('Data to be learned')
    pl.xlabel('<X, coef>')
    pl.ylabel('y')
    pl.show()

    # print the performance of ranking
    rank_svm = RankSVM().fit(X[train], Y[train])
    print 'Performance of ranking ', rank_svm.score(X[test], Y[test])

    # and that of linear regression
    ridge = linear_model.RidgeCV(fit_intercept=False)
    ridge.fit(X[train], y[train])
    X_test_trans, y_test_trans, diff = transform_pairwise(X[test], y[test])
    score = np.mean(np.sign(ridge.predict(X_test_trans)) == y_test_trans)
    print 'Performance of linear regression ', score
