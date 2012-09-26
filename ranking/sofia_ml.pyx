# distutils: language = c++

cdef extern from "src/sf-data-set.h":
    cdef cppclass SfDataSet:
        SfDataSet(bool)
        SfDataSet(string, int, bool)

def bla():
    return 0