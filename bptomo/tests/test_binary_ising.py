from ..bp_reconstruction import BP_step, BP_step_parallel, _initialize_field,\
                                    _calc_hatf
from ..build_projection_operator import build_projection_operator
from ..util import generate_synthetic_data
import numpy as np
from scipy import sparse


def test_full_reco_can():
    """
    Regression test: this example uses the canonical formulation
    """
    L = 64
    im = generate_synthetic_data(L)
    im -= 0.5
    im *= 2
    X, Y = np.ogrid[:L, :L]
    mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
    im[~mask] = 0
    # Build projection data
    n_dir = 16
    op = build_projection_operator(L, n_dir, mask=mask)
    y = (op * im[mask][:, np.newaxis]).ravel()
    sums = []
    op = sparse.lil_matrix(op)
    # Update after all measures
    h_m_to_px = _initialize_field(y, L, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    for i in range(6):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = \
                    BP_step(h_m_to_px, h_px_to_m, y, op, L, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in sums]
    assert err[-1] == 0
    # Use the Parallel algorithm
    """
    sums = []
    h_m_to_px = _initialize_field(y, L, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    for i in range(6):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = \
                    BP_step_parallel(h_m_to_px, h_px_to_m, y, op, L, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0)[mask]).sum() for sumi in sums]
    print err
    assert err[-1] == 0
    """
