# distutils: language = c++

BUFFER_MB = 40 # default in sofia-ml

cdef extern from "src/sf-data-set.h":
    cdef cppclass SfDataSet:
        SfDataSet(bool)
        SfDataSet(string, int, bool)

def train(train, fit_intercept=True):
    data = SfDataSet(train, BUFFER_MB, fit_intercept)
