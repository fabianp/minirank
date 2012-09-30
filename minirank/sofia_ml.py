import tempfile
import numpy as np
from sklearn import datasets
import _sofia_ml

def train(X, y, alpha, query_id, max_iter=100, model='rank', step_probability=0.5):
    """
    model : {'rank', 'combined-ranking'}

    """
    if query_id is None:
        query_id = np.ones(y.size)
    with tempfile.NamedTemporaryFile() as f:
        datasets.dump_svmlight_file(X, y, f.name, query_id=query_id)
        w = _sofia_ml.train(f.name, X.shape[1], alpha, max_iter, False, model,
            step_probability)
    return w
