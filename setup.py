from distutils.core import setup
import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import sys

if sys.platform == 'darwin':
    extra_link_args = ['-std=c++11','-stdlib=libc++']
    extra_compile_args = ['-std=c++11','-stdlib=libc++']
else:
    extra_link_args = ['-std=c++11']
    extra_compile_args = ['-std=c++11']

ext_modules = [
Extension("poreover.decoding.decoding_cpp", sources=["poreover/decoding/decoding_cpp.pyx"], include_dirs=[np.get_include()], language='c++',extra_link_args=extra_link_args, extra_compile_args=extra_compile_args),
Extension("poreover.decoding.decoding_cy", sources=["poreover/decoding/decoding_cy.pyx"], include_dirs=[np.get_include()], language='c++',extra_link_args=extra_link_args, extra_compile_args=extra_compile_args),
Extension("poreover.align.align", sources=["poreover/align/align.pyx"], include_dirs=[np.get_include()])
]

setup(
    name="poreover",
    version="0.0",
    packages=setuptools.find_namespace_packages(),
    ext_modules = cythonize(ext_modules, annotate=True),
    entry_points={'console_scripts':['poreover = poreover.__main__:main']}
)
