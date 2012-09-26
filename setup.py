from distutils.core import setup
from Cython.Build import cythonize

setup(name='ranking',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='',
    packages=['ranking'],
    ext_modules = cythonize('ranking/sofia_ml.pyx')
)