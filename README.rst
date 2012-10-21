Some ranking algorithms in Python.

Dependencies
------------

  - cython >= 0.17 (previous versions will not work)
  - numpy

Methods
-------

minirank.train_sgd

    Trains a model using stochastic gradient descent. See docstring for
    more details.

minirank.compat.RankSVM implements an estimator following the conventions
used in scikit-learn.