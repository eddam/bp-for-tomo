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

* optional: matplotlib (to plot the demo results)

* a C compiler 

Install
-------

If you wish to use the code inside the source directory:

$ python setup.py build_ext --inplace

Or if you wish to install the code as a package that you can import in
all your python files:

$ python setup.py install 

(this will put the Python package somewhere in your PYTHONPATH, but you
might need root access in Linux to do so)

You can also specify the installation directory :

$ python setup.py install --prefix=/path/to/my_dir


Demo
----

run

$ python demo_blobs.py

(or, better, run the script inside the Ipython interpreter)
