#!/usr/bin/env python

from numpy.distutils.core import setup
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)


    config.add_subpackage('bptomo')

    config.add_extension(
        'bptomo.tan_tan',
        sources=['bptomo/tan_tan.c'],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        'bptomo.solve',
        sources=['bptomo/solve.c'],
        include_dirs=[numpy.get_include()],
    )

    return config


if __name__ == "__main__":

    setup(configuration=configuration,
          name='bptomo',
          packages=['bptomo']
          )
