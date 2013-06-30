Belief-propagation for binary tomography reconstruction
=======================================================

This Python code implements belief-propagation iterations for solving the
tomography reconstruction problem for binary images with a spatial
regularization.

Download
--------

A zipball of the source code is available on
https://github.com/eddam/bp-for-tomo/zipball/demo

If you wish to download the whole versioned project, you can clone it
with git (http://git-scm.com/):

$ git clone https://github.com/eddam/bp-for-tomo.git

Dependencies
------------

to use this code, you will need the Python language, together with
several Python modules

* python >=2.6

* numpy >= 1.4

* scipy

* a C compiler 

* optional: matplotlib (to plot the demo results) and scikit-image (for
  utility functions to preprocess real data).

All these packages are included in the usual Scientific Python
distribution, such as Anaconda http://continuum.io/downloads or Enthought
Python Distribution https://www.enthought.com/products/epd/. 

Install
-------

Go to the bptomo directory

$ cd bptomo

And execute one of the following commands:

If you wish to use the code inside the source directory:

$ python setup.py build_ext --inplace

(this will build the compiled parts of the code, using cython).

Or you can install the package in order to import it from anywhere in the
system:

$ python setup.py install --user


Demo
----

run the script

$ python demo_bp_flavors.py

(or, better, run the script inside the Ipython interpreter)
