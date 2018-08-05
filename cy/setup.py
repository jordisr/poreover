from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("decoding_cy.pyx", annotate=True)
)
