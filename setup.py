from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [
Extension("decoding.decoding_cpp", sources=["decoding/decoding_cpp.pyx"], include_dirs=[np.get_include()], language='c++'),
Extension("decoding.decoding_cy", sources=["decoding/decoding_cy.pyx"], include_dirs=[np.get_include()], language='c++'),
#Extension("decoding.decoding2", sources=["decoding/decoding2.pyx"], include_dirs=[np.get_include()]),
#Extension("decoding.sparse", sources=["decoding/sparse.pyx"], include_dirs=[np.get_include()]),
Extension("align.align", sources=["align/align.pyx"], include_dirs=[np.get_include()])
]

setup(
    ext_modules = cythonize(ext_modules, annotate=True)
)
