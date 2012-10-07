import itertools
import numpy as np

def transform_pairwise(X, y, query_id=None):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem.

    In this method, all pairs are chosen, except for those that have the
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
    if query_id is None:
        query_id = np.ones(len(y))
    y = np.asarray(y)
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    for (i, j) in comb:
        #if np.abs(y[i, 0] - y[j, 0]) <= 1. or y[i, 1] != y[j, 1]:
        if y[i] == y[j] or query_id[i] != query_id[j]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        diff.append(y[i] - y[j])
        y_new.append(np.sign(diff[-1]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
            diff[-1] = - diff[-1]
        k += 1
    return np.asarray(X_new), np.asarray(y_new).ravel(), np.array(diff).ravel()