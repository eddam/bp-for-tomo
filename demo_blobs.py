import numpy as np
from scipy import sparse
from bptomo.bp_reconstruction import BP_step, _initialize_field, _calc_hatf, \
                            BP_step_update_direction
from bptomo.build_projection_operator import build_projection_operator
from bptomo.util import generate_synthetic_data
import matplotlib.pyplot as plt

# Generate synthetic data
L = 256
im = generate_synthetic_data(L, n_pts=100)
im -= 0.5
im *= 2

X, Y = np.ogrid[:L, :L]
mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
im[~mask] = 0


# Build projection data
n_dir = L / 5
op = build_projection_operator(L, n_dir)
y = (op * im.ravel()[:, np.newaxis]).ravel()
# Add some noise
np.random.seed(0)
y += 4*np.random.randn(*y.shape)
op = sparse.lil_matrix(op)

# Prepare fields
sums = [] # total magnetization

h_m_to_px = _initialize_field(y, op) # measure to pixel
h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
h_ext = np.zeros_like(y) # external field

px_to_m, m_to_px = [], []

err_measure = []

for i in range(20):
    print "iter %d" %i
    h_m_to_px, h_px_to_m, h_sum, h_ext = BP_step_update_direction(h_m_to_px,
                                    h_px_to_m, y, op, J=0.1, hext=h_ext)
    sums.append(h_sum)
    m_to_px.append(h_m_to_px)
    px_to_m.append(h_px_to_m)
    segmentation = np.sign(h_sum.reshape(L, L))
    segmentation[~mask] = 0
    #y_segmentation = (op * segmentation.ravel()[:, np.newaxis]).ravel()
    #err_measure.append(((y - y_segmentation)**2).sum())

err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
print("number of errors vs. iteration: ")
print(err)

flip = [np.abs(np.sign(sums[i]) - np.sign(sums[i+1])).sum() for i in range(21)]

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(sums[-1].reshape(-L, L), vmin=-10, vmax=10)
plt.axis('off')
plt.title('local magnetization')
plt.subplot(133)
plt.semilogy(err, 'o', ms=8)
#plt.semilogy(err_measure, 'o', ms=8)
plt.xlabel('$n$', fontsize=18)
plt.title('# of errors')

plt.show()
