"""
used metrics
"""

import numpy as np
import itertools


def pairwise_disagreement(y_true, y_pred):
    """
    Number of pairwise inversions in a totally ordered 
    linear list. 
    """
    comb = itertools.combinations(range(y_true.size), 2)
    count = 0
    diff = 0.
    idx = np.argsort(y_true)
    y_pred = y_pred[idx]
    for (i, j) in comb:
        count += 1
        assert i < j
        if y_pred[i] >= y_pred[j]:
            diff += 1
    return diff / count
