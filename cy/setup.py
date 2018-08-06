from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [Extension("decoding_cy", sources=["decoding_cy.pyx"], include_dirs=[np.get_include()])]

setup(
    ext_modules = cythonize(ext_modules, annotate=True)
)
