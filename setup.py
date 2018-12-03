from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

my_extra_compile_args = ['-fopenmp', '-ffast-math']
my_extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "fmm_tree_c",
        sources=["fmm_tree_c.pyx"],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "tree_build_routines_c",
        sources=["tree_build_routines_c.pyx"],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "list_build_routines_c",
        sources=["list_build_routines_c.pyx"],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "tree_compute_routines_c",
        sources=["tree_compute_routines_c.pyx"],
        libraries=["m"],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    )
]

setup(name="Tree_Compute_Routines_C",
    ext_modules = cythonize(ext_modules, gdb_debug=True),
    include_path = [np.get_include()]
)