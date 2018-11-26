from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("./*.pyx"),
    include_path = [np.get_include()]
)