"""
This example demonstrates how to use BP-tomo when the binary values of the two
phases are not perfectly known, using expectation-maximization iterations. This is done by alternating two steps:

* BP-tomo is solved for the given measures, assuming that the binary values
  are +1 and -1. This results in a binary segmentation of the image.

* Then this segmentation is used to compute the optimal binary values for
  observing the tomography measures, knowing the segmentation. (This is a simple  linear least squares optimization). The tomography measures are subsequently
  rescaled using these new values, in order to correspond to a binary image
  with +1 and -1.

In the following example, we start the reconstruction with a 20% error on the
binary values (1.2 instead of 1). In 4 iterations of the
expectation-maximization scheme, the true binary values are found to an
excellent precision, and an error-free reconstruction is achieved.
"""

print(__doc__)

import numpy as np
from scipy import sparse, linalg
from bptomo.bp_reconstruction import BP_step_asym, \
                            _initialize_field, _calc_hatf
from bptomo.build_projection_operator import build_projection_operator
from bptomo.util import generate_synthetic_data
from bptomo.realdata.preprocessing import rescale_sino_to_binary

def generate_data(L, n_dir, sigma=1, n_pts=100, binary_value=1):
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

    binary_value: float
        The binary values of the image are ``binary_value`` and 
        ``- binary_value``. Default values are -1 and 1, but is possible
        to specify a different value to test the expectation-maximization
        algorithm works.
    """
    # Generate synthetic binary data
    im = generate_synthetic_data(L, n_pts=n_pts)
    im -= 0.5
    im *= 2 * binary_value
    X, Y = np.ogrid[:L, :L]
    mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
    im[~mask] = 0  # we only consider pixels inside a central circle

    # Projection operator
    op = build_projection_operator(L, n_dir, mask=mask)

    # Build projection data with noise
    y = (op * im[mask][:, np.newaxis]).ravel()
    np.random.seed(0)
    y += sigma*np.random.randn(*y.shape)

    # lil sparse format is needed to retrieve indices efficiently
    op = sparse.lil_matrix(op)
    return im, mask, y, op


def solve_bp_tomo(y, L, op, mask, im=None, n_iter=30):
    """
    Parameters
    ----------

    y: 1-D ndarray
        tomography measures (raveled in a 1-D vector)

    L: int
        linear size of image to be reconstructed

    op: sparse linear operator
        projection operator corresponding to the measures

    mask: ndarray of size LxL
        mask of pixels to be reconstructed

    im: ndarray of size LxL, optional (default None)
        optional, ground truth of image to be reconstruted. If provided,
        the number of errors is computed.

    n_iter: int
        how many BP iterations to perform
    """
    sums_uncoupled = [] # total magnetization

    # Prepare fields
    h_m_to_px = _initialize_field(y, L, op) # measure to pixel
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px) # pixel to measure
    h_ext = np.zeros_like(y) # external field

    # BP iterations
    for i in range(n_iter):
        print "iteration %d / %d" %(i + 1, n_iter)
        h_m_to_px, h_px_to_m, h_sum, h_ext = BP_step_asym(h_m_to_px, h_px_to_m,
                                            y, op, L, hext=h_ext, J=2)
        sums_uncoupled.append(h_sum)

    if im is not None:
        # Compute segmentation error from ground truth
        err_uncoupled = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in
                                                    sums_uncoupled]
        print("number of errors vs. iteration: ")
        print(err_uncoupled)

    # Compute new binary values by least square optimization
    mask1 = sums_uncoupled[-1] > 0
    mask2 = sums_uncoupled[-1] < 0
    val1 = op * mask1[:, None]
    val2 = op * mask2[:, None]
    vals = np.hstack((val1, val2))
    phase_vals = linalg.lstsq(vals, y[:, None])[0]
    v_down, v_up = np.sort(phase_vals)
    print "optimal binary values for this segmentation: ", v_down, v_up
    sino = y.reshape((n_dir, L))

    # Update tomography measures using new values
    rescaled_sino, ls = rescale_sino_to_binary(sino, v_down, v_up, op=op)
    y = rescaled_sino.ravel()
    return y, sums_uncoupled[-1]


L, n_dir, sigma, n_pts = 128, 40, 1, 100
im, mask, y, op = generate_data(L, n_dir, sigma, n_pts, binary_value=1.2)
n_iter = 30

# ------------------ Uncoupled lines -------------------------------------

for i in range(4):
    print "EM iteration No ", i
    y, res = solve_bp_tomo(y, L, op, mask, im=im, n_iter=30)
