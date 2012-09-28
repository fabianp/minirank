from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import glob
import numpy as np

sources =['ranking/_sofia_ml.pyx'] + glob.glob('ranking/src/*.cc')

setup(name='ranking',
    version='0.1',
    description='Ranking algorithms',
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='',
    packages=['ranking'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('ranking._sofia_ml',
        sources=sources,
        language='c++', include_dirs=[np.get_include()])],
)



#modules_list = create_extension_list("*_cpp.pyx",exclude=["*_c.pyx"])
#for module in modules_list:
#    module.language = "c++"
#ext_modules = cythonize(pyx_modules)