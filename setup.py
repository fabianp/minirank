from distutils.core import setup
from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list

setup(name='ranking',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='',
    packages=['ranking'],
    ext_modules = cythonize(['ranking/_sofia_ml.pyx'])
)



#modules_list = create_extension_list("*_cpp.pyx",exclude=["*_c.pyx"])
#for module in modules_list:
#    module.language = "c++"
#ext_modules = cythonize(pyx_modules)