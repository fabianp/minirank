import numpy as np

from ranking import train

np.random.seed(0)
X = np.random.randn(20, 5)
w = np.random.randn(5)
y = np.dot(X, w)
w_ = train(X, y, 1., np.ones(20))

assert np.allclose(np.argsort(y), np.argsort(np.dot(X, w)))
