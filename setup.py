#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(
    name='bptomo',
    packages=['bptomo'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("bptomo._ising", ["bptomo/_ising.pyx"]),
                    Extension("bptomo.tan_tan", ["bptomo/tan_tan.pyx"])]
    )
