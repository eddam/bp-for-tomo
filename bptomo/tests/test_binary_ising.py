from ..solve_ising import solve_microcanonical_h, \
                          gaussian_weight
from .._ising import solve_microcanonical_chain
from ..bp_reconstruction import BP_step, BP_step_update_direction, \
        BP_step_parallel, BP_step_always_update, _initialize_field, _calc_hatf
from ..build_projection_operator import build_projection_operator
from ..util import generate_synthetic_data
import numpy as np
from scipy import sparse

def microcanonical_bf(h, J, s0):
    """
    brute force method for 3 spins
    """
    # All possible configurations of 3 spins
    spins = np.mgrid[-1:2:2, -1:2:2, -1:2:2]
    spins.shape = (3, 8)
    spins = spins.T
    # minus energy
    minusH = (h * spins).sum(axis=1) + \
                J * (spins[:, 1:] * spins[:, :-1]).sum(axis=1)
    proba_config = np.exp(minusH) * gaussian_weight(spins.sum(axis=1), s0, beta=10)
    proba_spin = np.zeros((2, 3))
    for s in range(3):
        proba_spin[0, s] = (proba_config[spins[:, s] == -1]).sum()
        proba_spin[1, s] = (proba_config[spins[:, s] == 1]).sum()
    res = 0.5 * (np.log(proba_spin[1]) - np.log(proba_spin[0]))
    res[res > 10] = 10
    res[res < -10] = -10
    return res

def test_compare_bf_transfmatrix():
    mag_tmat = solve_microcanonical_h([0, 0, 0], [0.1, 0.1, 0.1], 1)
    mag_bf = microcanonical_bf([0, 0, 0], 0.1, 1)
    assert np.allclose(mag_bf, mag_tmat, rtol=1.e-2)
    mag_tmat = solve_microcanonical_h([1, 1, -1], [0.1, 0.1, 0.1], 1)
    mag_bf = microcanonical_bf([1, 1, -1], 0.1, 1)
    assert np.allclose(mag_bf, mag_tmat, rtol=1.e-2)
    # Works with large values as well
    mag_tmat = solve_microcanonical_h([10, 10, -10], [0.1, 0.1, 0.1], 1)
    mag_bf = microcanonical_bf([10, 10, -10], 0.1, 1)
    assert np.allclose(mag_bf, mag_tmat, rtol=1.e-2)
    mag_tmat = solve_microcanonical_h([10, 10, -10], [0.1, 0.1, 0.1], -1)
    mag_bf = microcanonical_bf([10, 10, -10], 0.1, -1)
    assert np.allclose(mag_bf, mag_tmat, rtol=1.e-2)


def test_solve_chain():
    h = np.ones(10)
    h[::2] = -1
    # First example where we can be aligned with all local fields
    proba = solve_microcanonical_chain(h, 0.2*np.ones(10), 0)
    res = np.ones(10, dtype=bool)
    res[::2] = False
    assert np.all((proba[1] > proba[0]) == res)
    h[:2] *= 0.1
    # Here we must have some misalignment, check that it happens
    # for the lowest absolute value of the local field
    proba = solve_microcanonical_chain(h, 0.2*np.ones(10), 2)
    assert np.all((proba[1] > proba[0])[:2])
    # Case where the coupling J is more important
    # the amplitude of h is increasing from left to right
    h = np.ones(10)
    h[::2] = -1
    h *= np.linspace(0.1, 0.8, 10)
    proba = solve_microcanonical_chain(h, 0.2*np.ones(10), 0)
    assert np.all((proba[1] > proba[0]) == res)
    res_two_blocks = np.ones(10, dtype=bool)
    res_two_blocks[:5] = False
    proba = solve_microcanonical_chain(h, 2*np.ones(10), 0)
    assert np.all((proba[1] > proba[0]) == res_two_blocks)
    # 2 blocks of disconnected spins 
    J = np.ones(10)
    J[3] = 0
    h = np.zeros(10)
    proba = solve_microcanonical_chain(h, J, 2)
    assert set(np.argsort(proba[1] / proba[0])[:4]) == set([0, 1, 2, 3])


def test_full_reco_microcan():
    """
    Regression test: a real-life example
    """
    L = 16
    # Build image
    im = -1 * np.ones((L, L))
    im[6:12, 6:12] = 1
    im[9, 9:12] = -1
    n_dir = 8
    X, Y = np.ogrid[:L, :L]
    mask = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
    im[~mask] = 0
    # Operator
    op = build_projection_operator(L, n_dir)
    # Measurements (noise-free)
    y = (op * im.ravel()[:, np.newaxis]).ravel()
    op = sparse.lil_matrix(op)
    h_m_to_px = _initialize_field(y, op, big_field=10.)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    sums = []
    for i in range(6):
        h_m_to_px, h_px_to_m, h_sum, hext = BP_step(h_m_to_px, h_px_to_m, y,
                                        op, use_micro=True)
        sums.append(h_sum)
    # Check that segmentation is correct
    assert np.all((sums[2].reshape((L, L)) > 0) == (im > 0))
    # Check that rapidly, all spins are blocked
    assert np.all(np.logical_or(sums[-1][mask.ravel()]<-8,
                                        sums[-1][mask.ravel()]>8))

def test_full_reco_can():
    """
    Regression test: this larger example uses the canonical formulation
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
    op = build_projection_operator(L, n_dir)
    y = (op * im.ravel()[:, np.newaxis]).ravel()
    sums = []
    op = sparse.lil_matrix(op)
    # Update after all measures
    h_m_to_px = _initialize_field(y, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    for i in range(6):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = \
                    BP_step(h_m_to_px, h_px_to_m, y, op, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
    assert err[-1] == 0
    # Use the Parallel algorithm
    sums = []
    h_m_to_px = _initialize_field(y, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    for i in range(6):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = \
                    BP_step_parallel(h_m_to_px, h_px_to_m, y, op, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
    print err
    assert err[-1] == 0
    # Update after every direction
    h_m_to_px = _initialize_field(y, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    for i in range(3):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = BP_step_update_direction(h_m_to_px,
                                h_px_to_m, y, op, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
    assert err[-1] == 0
    print err
    # Update after every ray
    h_m_to_px = _initialize_field(y, op)
    h_px_to_m, first_sum = _calc_hatf(h_m_to_px)
    hext = np.zeros_like(y)
    sums = []
    for i in range(3):
        print "iter %d" %i
        h_m_to_px, h_px_to_m, h_sum, hext = BP_step_always_update(h_m_to_px,
                                h_px_to_m, y, op, hext=hext)
        sums.append(h_sum)
    err = [np.abs((sumi>0) - (im>0).ravel()).sum() for sumi in sums]
    assert err[-1] == 0
    print err
