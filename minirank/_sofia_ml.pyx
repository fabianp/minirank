# distutils: language = c++
# distutils: sources = minirank/src/sofia-ml-methods.cc ranking/src/{sf-weight-vector.cc,sf-data-set.cc}
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

BUFFER_MB = 40 # default in sofia-ml

cdef extern from "src/sofia-ml-methods.h":
    cdef cppclass SfDataSet:
        SfDataSet(bool)
        SfDataSet(string, int, bool)

    cdef cppclass SfWeightVector:
        SfWeightVector(int)
        SfWeightVector(string)
        string AsString()
        float ValueOf(int)

cdef extern from "src/sofia-ml-methods.h" namespace "sofia_ml":
    cdef enum LearnerType:
        PEGASOS, MARGIN_PERCEPTRON, PASSIVE_AGGRESSIVE, LOGREG_PEGASOS,
        LOGREG, LMS_REGRESSION, SGD_SVM, ROMMA

    cdef enum EtaType:
        BASIC_ETA
        PEGASOS_ETA
        CONSTANT

    void StochasticRocLoop(SfDataSet, LearnerType, EtaType,
                           float, float, int, SfWeightVector*)

    void BalancedStochasticOuterLoop(SfDataSet, LearnerType, EtaType,
                                     float, float, int, SfWeightVector*)

    void StochasticRankLoop(SfDataSet, LearnerType, EtaType,
          float, float, int, SfWeightVector*)

    void StochasticClassificationAndRankLoop(SfDataSet, LearnerType, EtaType,
        float, float, float, int num_iters, SfWeightVector*)

    void SvmPredictionsOnTestSet(SfDataSet test_data,
        SfWeightVector, vector[float]*)

def train(train_data, int n_features, float alpha, int max_iter, bool fit_intercept,
          model, float step_probability):
    cdef SfDataSet *data = new SfDataSet(train_data, BUFFER_MB, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(n_features)
    cdef float c = 0.0
    cdef int i
    if model == 'rank':
        StochasticRankLoop(deref(data), PEGASOS, BASIC_ETA, alpha, c, max_iter, w)
    elif model == 'roc':
        StochasticRocLoop(deref(data), SGD_SVM, BASIC_ETA, alpha, c, max_iter, w)
    elif model == 'combined-ranking':
        StochasticClassificationAndRankLoop(deref(data), SGD_SVM, BASIC_ETA, alpha, c,
            step_probability, max_iter, w)
    else:
        raise NotImplementedError
    cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
    for i in range(n_features):
        coef[i] = w.ValueOf(i)
    return coef


def predict(test_data, string coef, bool fit_intercept):
    cdef SfDataSet *test_dataset = new SfDataSet(test_data, BUFFER_MB, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(coef)
    cdef vector[float] *predictions = new vector[float]()
    SvmPredictionsOnTestSet(deref(test_dataset), deref(w), predictions)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] out = np.empty(predictions.size())
    for i in range(predictions.size()):
        out[i] = predictions.at(i)
    return out
