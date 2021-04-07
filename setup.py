from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name="Melody selection",
    ext_modules=cythonize("./salience/salience.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
