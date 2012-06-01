import numpy as np
from bp_reconstruction import BP_step, _initialize_field, _calc_hatf
from build_projection_operator import build_projection_operator
from util import generate_synthetic_data


L = 128
im = generate_synthetic_data(L)
im -= 0.5
im *= 2

X, Y = np.ogrid[:L, :L]
mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
im[~mask] = 0


# Build projection data
n_dir = 36
op = build_projection_operator(L, n_dir)
y = (op * im.ravel()[:, np.newaxis]).ravel()
y += 8*np.random.randn(*y.shape)

m_to_px, px_to_m, sums = [], [], []

h_m_to_px = _initialize_field(y, op)
h_px_to_m, first_sum = _calc_hatf(h_m_to_px)

first_m_to_px = np.copy(h_m_to_px)

#h_m_to_px = np.zeros_like(h_m_to_px)
#h_px_to_m = np.zeros_like(h_px_to_m)


#h_px_to_m, h_m_to_px = np.zeros((2, n_dir, im.size))
for i in range(20):
    print "iter %d" %i
    h_m_to_px, h_px_to_m, h_sum = BP_step(h_m_to_px, h_px_to_m, y, op, damping=0.98, J=1)
    m_to_px.append(h_m_to_px)
    px_to_m.append(h_px_to_m)
    sums.append(h_sum)
