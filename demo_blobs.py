"""
This example shows the reconstruction of a binary image from a reduced set
of tomographic projections.

Synthetic binary data of size 128x128 are generated, and 30 projections
along regularly-spaced angles are computed.

We first compute the initial guess for the field h_m_to_px (message passing
from measures to pixels), then we perform belief-propagation iterations. The
segmentation error with respect to ground truth decreases very fast
(exponentially) with BP iterations, until an exact reconstruction is reached
in this example.
"""

print(__doc__)

import numpy as np
from scipy import sparse
from bptomo.bp_reconstruction import BP_step, BP_step_mf, \
                            _initialize_field, _calc_hatf_mf, _calc_hatf
from bptomo.build_projection_operator import build_projection_operator
from bptomo.util import generate_synthetic_data
import matplotlib.pyplot as plt
from time import time

# Generate synthetic binary data (pixels values in {-1, 1})
L = 128
im = generate_synthetic_data(L, n_pts=100)
im -= 0.5
im *= 2

X, Y = np.ogrid[:L, :L]
mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
im[~mask] = 0  # we only consider pixels inside a central circle


# Build projection data with noise
n_dir = 30
op = build_projection_operator(L, n_dir)

y = (op * im.ravel()[:, np.newaxis]).ravel()
# Add some noise
np.random.seed(0)
y += 2*np.random.randn(*y.shape)

# lil sparse format is needed to retrieve indices efficiently
op = sparse.lil_matrix(op)


n_iter = 5

# ------------------ Exact -------------------------------------

# Prepare fields
sums = [] # total magnetization

h_m_to_px = _initialize_field(y, op) # measure to pixel
h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
h_sum = _calc_hatf_mf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

px_to_m, m_to_px = [], []


t0 = time()

for i in range(n_iter):
    print "iteration %d / %d" %(i + 1, n_iter)
    h_m_to_px, h_px_to_m, h_sum, h_ext = BP_step(h_m_to_px, h_px_to_m, y, op, hext=h_ext)
    sums.append(h_sum)
    m_to_px.append(h_m_to_px)

t1 = time()
print "reconstruction done in %f s" %(t1 - t0)

# Compute segmentation error from ground truth
err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
print("number of errors vs. iteration: ")
print(err)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(sums[-1].reshape(-L, L), vmin=-10, vmax=10,
                    interpolation='nearest')
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('# of errors')

plt.show()
# ------------------ Mean_field -------------------------------------

# Prepare fields
sums = [] # total magnetization

h_m_to_px = _initialize_field(y, op) # measure to pixel
h_sum = _calc_hatf_mf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

px_to_m, m_to_px = [], []


t0 = time()

for i in range(50):
    print "iteration %d / %d" %(i + 1, n_iter)
    h_m_to_px, h_sum, h_ext = BP_step_mf(h_m_to_px, h_sum, y, op, hext=h_ext, mf_correct=True)
    sums.append(h_sum)
    m_to_px.append(h_m_to_px)

t1 = time()
print "reconstruction done in %f s" %(t1 - t0)

# Compute segmentation error from ground truth
err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
print("number of errors vs. iteration: ")
print(err)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(sums[-1].reshape(-L, L), vmin=-10, vmax=10,
                    interpolation='nearest')
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('# of errors')

plt.show()
