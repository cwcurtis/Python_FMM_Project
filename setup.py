debug = True

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults

import numpy as np

numpy_include_dir = np.get_include()

#cy_options = {
#    'annotate': True,
#    'compiler_directives': {
#        'profile': debug,
#        'linetrace': debug,
#        'boundscheck': debug,
#        'wraparound': debug,
#        'initializedcheck': debug,
#        'language_level': 3,
#    },
#}

get_directive_defaults()['linetrace'] = True
get_directive_defaults()['binding'] = True

my_extra_compile_args = ['-fopenmp', '-ffast-math', '-O3']
my_extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "fmm_tree_c",
        sources=["fmm_tree_c.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "tree_build_routines_c",
        sources=["tree_build_routines_c.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "list_build_routines_c",
        sources=["list_build_routines_c.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    ),
    Extension(
        "tree_compute_routines_c",
        sources=["tree_compute_routines_c.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        libraries=["m"],
        extra_compile_args=my_extra_compile_args,
        extra_link_args=my_extra_link_args
    )
]

setup(name="Tree_Compute_Routines_C",
    ext_modules = cythonize(ext_modules)
)