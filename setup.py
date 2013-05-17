from distutils.core import setup
import numpy as np


setup(name='minirank',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='f@fabianp.net',
    url='',
    packages=['minirank'],
    cmdclass = {'build_ext': build_ext},
    requires = ['numpy', 'scipy'],
)
