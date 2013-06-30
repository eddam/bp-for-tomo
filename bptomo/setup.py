from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("_solve_lines", ["_solve_lines.pyx"],
                        include_dirs=[numpy.get_include()])]

setup(
  name = 'bptomo',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

