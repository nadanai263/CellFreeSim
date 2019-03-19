from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy 

### Run command: 
### python3 setup.py build_ext --inplace 

setup(
    ext_modules=cythonize((Extension("model", sources=["model.pyx"], 
    	include_dirs=[numpy.get_include()],),))
)

