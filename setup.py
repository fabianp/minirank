from distutils.core import setup
import numpy as np
import setuptools

setup(name='minirank',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='f@fabianp.net',
    url='',
    packages=['minirank'],
    requires = ['numpy', 'scipy'],
)
