from distutils.core import setup

setup(name='ranking',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='',
    py_modules=['ranksvm', 'isotron', 'isotonic_regression', 'datasets'],
)