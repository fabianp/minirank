from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import glob
import numpy as np


setup(name='ranking',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='',
    packages=['minirank'],
    cmdclass = {'build_ext': build_ext},
    requires = ['numpy', 'scipy'],
)