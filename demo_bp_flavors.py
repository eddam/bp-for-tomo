"""

This example shows the reconstruction of a binary image from a reduced set of
tomographic projections. Synthetic binary data of size 128x128 are generated,
and 40 projections along regularly-spaced angles are computed.

Different versions of the BP-tomo algorithm are compared:

 - the uncoupled version: interactions between neighboring spins are used
   only for four directions. On the other angles, we just impose that the
   sum of magnetizations corresponds to the tomographic measurement. This
   is the fastest method (the difference is even greater for a large image
   size).

 - the coupled version: spins are coupled on all lines, and an Ising chain is
   solved for each line at each iteration. Note that different
   values of the coupling factor J have to be used for the coupled and
   uncoupled versions!

 - the mean field version: pixels pass the same value to all factors to which
   they belong.

All versions are supposed to give similar results. The mean field version
gives better results for a large number of measurements, where the mean field
approximation is very good.

You may play on the parameters `L` (linear size of the image), `n_dir` (number
of angles) and `n_pts` (the size of structures is proportional to
1/sqrt(n_pts)).
"""

print(__doc__)

import numpy as np
from scipy import sparse
from bptomo.bp_reconstruction import BP_step, BP_step_asym, BP_step_mf, \
                            _initialize_field, _calc_hatf_mf, _calc_hatf
from bptomo.build_projection_operator import build_projection_operator
from bptomo.util import generate_synthetic_data
import matplotlib.pyplot as plt
from time import time

def generate_data(L, n_dir, sigma=1, n_pts=100):
    """
    Parameters
    ----------

    L: int
        linear size of the image

    n_dir: int
        number of angles

    sigma: float
        absolute intensity of Gaussian noise added on projections

    n_pts: int
        Parameter used to tune the size of structures in the image.
    """
    # Generate synthetic binary data (pixels values in {-1, 1})
    im = generate_synthetic_data(L, n_pts=n_pts)
    im -= 0.5
    im *= 2

    X, Y = np.ogrid[:L, :L]
    mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
    im[~mask] = 0  # we only consider pixels inside a central circle

    # Build projection data with noise
    op = build_projection_operator(L, n_dir, mask=mask)

    y = (op * im[mask][:, np.newaxis]).ravel()
    # Add some noise
    np.random.seed(0)
    y += sigma*np.random.randn(*y.shape)

    # lil sparse format is needed to retrieve indices efficiently
    op = sparse.lil_matrix(op)
    return im, mask, y, op


L, n_dir, sigma, n_pts = 128, 40, 1, 100
im, mask, y, op = generate_data(L, n_dir, sigma, n_pts)
n_iter = 18

# ------------------ Uncoupled lines -------------------------------------

# Prepare fields
sums_uncoupled = [] # total magnetization

h_m_to_px = _initialize_field(y, L, op) # measure to pixel
h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

t0 = time()

for i in range(n_iter):
    print "iteration %d / %d" %(i + 1, n_iter)
    h_m_to_px, h_px_to_m, h_sum, h_ext = BP_step_asym(h_m_to_px, h_px_to_m,
                                        y, op, L, hext=h_ext, J=2)
    sums_uncoupled.append(h_sum)


t1 = time()
print "uncoupled lines: reconstruction done in %f s" %(t1 - t0)

# Compute segmentation error from ground truth
err_uncoupled = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in
                                                sums_uncoupled]
print("number of errors vs. iteration: ")
print(err_uncoupled)


res = np.zeros_like(im)
res[mask] = sums_uncoupled[-1]

# Plot result
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(res, vmin=-10, vmax=10,
                    interpolation='nearest')
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err_uncoupled, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('uncoupled lines: # of errors')

plt.show()

# ------------------ Coupled lines -------------------------------------

# Prepare fields
sums_coupled = [] # total magnetization

h_m_to_px = _initialize_field(y, L, op) # measure to pixel
h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

t0 = time()

for i in range(n_iter):
    print "iteration %d / %d" %(i + 1, n_iter)
    h_m_to_px, h_px_to_m, h_sum, h_ext = BP_step(h_m_to_px, h_px_to_m,
                                        y, op, L, hext=h_ext)
    sums_coupled.append(h_sum)

t1 = time()
print "coupled lines: reconstruction done in %f s" %(t1 - t0)

# Compute segmentation error from ground truth
err_coupled = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in sums_coupled]
print("number of errors vs. iteration: ")
print(err_coupled)


res = np.zeros_like(im)
res[mask] = sums_coupled[-1]

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(res, vmin=-10, vmax=10,
                    interpolation='nearest')
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err_coupled, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('coupled lines: # of errors')

plt.show()

# ------------------ Mean field -------------------------------------

# Prepare fields
sums_mf = [] # total magnetization

h_m_to_px = _initialize_field(y, L, op) # measure to pixel
h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
h_sum = _calc_hatf_mf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

t0 = time()

for i in range(n_iter):
    print "iteration %d / %d" %(i + 1, n_iter)
    h_m_to_px, h_sum, h_ext = BP_step_mf(h_m_to_px, h_sum, y, op, L, hext=h_ext)
    sums_mf.append(h_sum)

t1 = time()
print "coupled lines: reconstruction done in %f s" %(t1 - t0)

# Compute segmentation error from ground truth
err_mf = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in sums_mf]
print("number of errors vs. iteration: ")
print(err_mf)


res = np.zeros_like(im)
res[mask] = sums_mf[-1]

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(res, vmin=-10, vmax=10,
                    interpolation='nearest')
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err_mf, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('mean field: # of errors')

plt.show()
