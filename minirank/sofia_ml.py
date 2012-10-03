import io, sys, tempfile
import numpy as np
from sklearn import datasets
import _sofia_ml

if sys.version_info[0] < 3:
    bstring = basestring
else:
    bstring = str

def train(X, y, alpha, query_id=None, max_iter=100, model='rank', step_probability=0.5):
    """
    model : {'rank', 'combined-ranking', 'roc'}

    Returns
    -------
    coef
    """
    if isinstance(X, bstring):
        if y is not None:
            y = int(y)
        else:
            y = 2 ** 17 # the default in sofia-ml
        w = _sofia_ml.train(X, y, alpha, max_iter, False, model,
            step_probability)
    else:
        X = np.asarray(X)
        if query_id is None:
            query_id = np.ones(X.shape[0])
        with tempfile.NamedTemporaryFile() as f:
            datasets.dump_svmlight_file(X, y, f.name, query_id=query_id)
            w = _sofia_ml.train(f.name, X.shape[1], alpha, max_iter, False, model,
                step_probability)
    return w, None

def predict(X, coef, query_id=None):
    s_coef = ''
    for e in coef:
        s_coef += '%.5f ' % e
    s_coef = s_coef[:-1]
    if isinstance(X, bstring):
        return _sofia_ml.predict(X, s_coef, False)
    else:
        X = np.asarray(X)
        if query_id is None:
            query_id = np.ones(X.shape[0])
        with tempfile.NamedTemporaryFile() as f:
            y = np.ones(X.shape[0])
            datasets.dump_svmlight_file(X, y, f.name, query_id=query_id)
            prediction = _sofia_ml.predict(f.name, s_coef, False)
        return prediction