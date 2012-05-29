import numpy as np
from bp_reconstruction import BP_step, _initialize_field, _calc_hatf
from build_projection_operator import build_projection_operator
from util import generate_synthetic_data


L = 32
"""
im = -1 * np.ones((L, L))
im[12:24, 12:24] = 1
im[18, 18:24] = -1
# Additional images
im[:10, :10] = 1
im[10:20, 25:] = 1
im[14:17, 26:28] = -1
#im[6:8, 10:14] = 1
"""
im = generate_synthetic_data(L)
im -= 0.5
im *= 2

X, Y = np.ogrid[:L, :L]
mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
im[~mask] = 0


# Build projection data
n_dir = 12
op = build_projection_operator(L, n_dir)
y = (op * im.ravel()[:, np.newaxis]).ravel()
#y += 0.1*np.random.randn(*y.shape)

m_to_px, px_to_m, sums = [], [], []

h_m_to_px = _initialize_field(y, op)
h_px_to_m, first_sum = _calc_hatf(h_m_to_px)

first_m_to_px = np.copy(h_m_to_px)

#h_m_to_px = np.zeros_like(h_m_to_px)
#h_px_to_m = np.zeros_like(h_px_to_m)


#h_px_to_m, h_m_to_px = np.zeros((2, n_dir, im.size))
for i in range(10):
    print "iter %d" %i
    h_m_to_px, h_px_to_m, h_sum = BP_step(h_m_to_px, h_px_to_m, y, op)
    m_to_px.append(h_m_to_px)
    px_to_m.append(h_px_to_m)
    sums.append(h_sum)
