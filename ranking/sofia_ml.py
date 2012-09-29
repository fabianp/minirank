import tempfile
from sklearn import datasets
import _sofia_ml

def train(X, y, alpha, query_id, model='rank_svm'):
    """
    bla bla
    """
    if model == 'rank_svm':
        with tempfile.NamedTemporaryFile() as f:
            datasets.dump_svmlight_file(X, y, f.name, query_id=query_id)
            w = _sofia_ml.train(f.name, X.shape[1], alpha)
    else:
        raise NotImplementedError
    return w
