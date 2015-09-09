import itertools
import numpy as np

def transform_pairwise(X, y, blocks=None):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem.

    In this method only pairs from the same block value are selected,
    except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        Target labels.
    b : array, shape (n_samples,), optional

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
    if blocks is None:
        blocks = np.ones(len(y))
    y = np.asarray(y)
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    for (i, j) in comb:
        if y[i] == y[j] or blocks[i] != blocks[j]:
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