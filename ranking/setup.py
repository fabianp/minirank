from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='ranking',
    ext_modules = cythonize('*.pyx'),
)