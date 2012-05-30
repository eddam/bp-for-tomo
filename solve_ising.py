import numpy as np
from scipy import ndimage, stats
import math
from math import exp, log1p
from scipy import stats
from _ising import _build_logZp, log_exp_plus_exp, solve_microcanonical_chain_pyx, solve_microcanonical_chain 
from tan_tan import fast_mag_chain, fast_mag_chain_nu

#-------------------------- Canonical formulation ----------------

def mag_chain(h, J, hext, full_output=False):
    """
    Compute the total magnetization for an Ising chain.
    Use the cython function fast_mag_chain_nu

    Parameters
    ----------

    h: 1-d ndarray of floats
        local field

    J: 1-d ndarray
        couplings between spins

    hext: float
        global field
    """
    magtot, hloc = fast_mag_chain_nu(h, J, hext)
    if full_output:
        return magtot, hloc
    else:
        return magtot

def solve_canonical_h(h, J, y):
    """
    Solve Ising chain in the canonical formulation.

    Parameters
    ----------

    h: 1-d ndarray of floats
        local field

    y: total magnetization

    J: 1-d ndarray of floats
        coupling between spins

    Returns
    -------

    hloc: 1-d ndarray of floats
        local magnetization
    """
    epsilon=.0001
    hext = 0
    N = len(h) 
    mag_tot = mag_chain(h, J, hext)
    if mag_tot < y:
        hmin = 0
        hext = 8
        while y - mag_chain(h, J, hext) > epsilon:
            hmin = hext
            hext *= 2
        hmax = hext
    else:
        hmax = 0
        hext = -8
        while mag_chain(h, J, hext) - y > epsilon:
            hmax = hext
            hext *= 2
        hmin = hext
    mag_tot = 2 * N
    iter_nb = 0
    # dichotomy
    while abs(mag_tot - y) / N > epsilon and iter_nb < 200:
        iter_nb += 1
        hext = 0.5 * (hmin + hmax)
        mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
        if mag_tot < y:
            hmin = hext
        else:
            hmax = hext
    return hloc


# ---------------------------------------------------------------

min_inf = -10000
max_inf = 500

#----------------Microconical Ising chain -----------------------------


def _build_left_right(h, J):
    """
    Build two partial partition functions, left and right.

    For computing the right partition function Tp, we just reverse the order
    of h and J, and compute the left partition function, and then reverse its
    order.
    """
    h = np.asarray(h).astype(np.float)
    J = np.asarray(J).astype(np.float)
    Zp = _build_logZp(h, J)
    Tp = _build_logZp(h[::-1], J[::-1])
    Tp = Tp[::-1]
    return Zp, Tp

def log_gaussian_weight(s, s0, beta=4.):
    """
    probability of s if the measure if s_0
    With the hypothesis of Gaussian white noise, it is a Gaussian.

    Parameters
    ----------

    s: float
        sum of spins

    s0: float
        measure

    beta: float
        width of the Gaussian. The more noise on the projections, the
        larger beta should be.
    """
    return np.maximum(-40, - beta * (s - s0)**2)


def gaussian_weight(s, s0, beta=4.):
    """
    probability of s if the measure if s_0
    With the hypothesis of Gaussian white noise, it is a Gaussian.

    Parameters
    ----------

    s: float
        sum of spins

    s0: float
        measure

    beta: float
        width of the Gaussian. The more noise on the projections, the
        larger beta should be.
    """
    return np.exp(np.maximum(-40, - beta * (s - s0)**2))

def solve_microcanonical_chain_old(h, J, s0, error=2):
    """
    Solve Ising chain for N spins, in the microcanonical formulation

    Parameters
    ----------
    h: 1-d ndarray of length N
        local field

    J: 1-d ndarray of length N
        local coupling between spin

    s0: float
        expected sum of spins

    error: int
        expected error on the projections.

    Returns
    -------
    proba: 2xN array
        proba[i, n] is the (not normalized) probability of spin n to
        be s_i

    Examples
    --------
    """
    N = len(h)
    prob = min_inf * np.ones((2, N))
    Zp, Tp = _build_left_right(h, J)
    # indices for the sum of spins
    u = np.arange(2 * N + 1)
    # Us is the value of the magnetization
    Us = u - N
    # Now we write the probability of spin i, which is given by
    # proba[s_i] = sum_{u, v, s_{n-1}, s_{n+1}}
    #         Z_{i-1}(u,s_{n-1}) T_{n+1}(v,s_{n+1}) x
    #        exp[J(s_{n-1}s_n+s_{n+1}s_n] exp[h_n s_n] w(s_n+u+v)
    for si in range(0, N):
        for uu, UUs in zip(u, Us):
            v = np.arange(s0 + 2*N - uu -1, s0 + 2*N -uu + 2., 2)
            v = v[np.logical_and(v >= 0, v < 2*N + 1)]
            for vv in v:
                s_n = -1
                prob[0, si] = log_exp_plus_exp(prob[0, si],
                    Zp[si, uu, 0] + Tp[si, vv, 0] \
                    - h[si] * s_n  \
                    + log_gaussian_weight(- s_n + UUs + vv - N, s0))
                s_n = 1
                prob[1, si] = log_exp_plus_exp(prob[1, si],
                    Zp[si, uu, 1] + Tp[si, vv, 1]  \
                    - h[si] * s_n  \
                    + log_gaussian_weight(- s_n + UUs + vv - N, s0))
    return np.exp(prob)

def solve_microcanonical_h(h, J, s0, error=1):
    """
    Compute local magnetization for microcanonical Ising chain
    """
    h = np.asarray(h).astype(np.float)
    J = np.asarray(J).astype(np.float)
    s0 = float(s0)
    try:
        proba = solve_microcanonical_chain(h, J, s0)
    except FloatingPointError:
        print 'other'
        proba = solve_microcanonical_chain_pyx(h, J, s0)
    res = 0.5 * (proba[1] - proba[0])
    big_field = 10
    res[res > big_field] = big_field
    res[res < -big_field] = -big_field
    return res


# ------------------ Solving Ising model for one projection -----------

def solve_line(field, Js, y, onsager=1, big_field=10, verbose=False):
    """
    Solve Ising chain

    Parameters
    ----------

    field: 1-d ndarray
        Local field on the spins

    y: float
        Sum of the spins (value of the projection)

    J: float
        Coupling between spins

    mask_res: 1-d ndarray of bools, same shape as field
        Mask of already blocked spins. mask_res is equal to +1 or -1
        for blocked spins, and 0 for spins to be determined.

    onsager: float

    big_field: float
        FIXME: do we need big_field here

    Returns
    -------

    hloc: 1-d ndarray, same shape as field
        local magnetization
    """
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if np.all(mask_blocked) and np.abs(np.sign(field).sum() - y) < 0.1:
        return (1.5 - onsager) * field
    elif False: #np.sum(~mask_blocked) > 25:
        hloc = solve_canonical_h(field, Js, y)
        mask_blocked = np.abs(hloc) > big_field
        hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
        hloc -= onsager * field
        return hloc
    else:
        if verbose:
            print "new chain"
            print field, Js, y
        hloc = solve_microcanonical_h(field, Js, y)
        mask_blocked = np.abs(hloc) > big_field
        hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
        if verbose:
            print hloc
        hloc -= onsager * field
        if verbose:
            print hloc
        return hloc


# ---------------- Initialization ------------------------------

def initialize_field(y, l_x, big_field=10):
    h_m_to_px = np.zeros((len(y), l_x))
    for dr in range(ndir):
        ratio = y[dr]/float(L)
        ratio = np.maximum(-1 + 1.e-15, ratio)
        ratio = np.minimum(1 - 1.e-15, ratio)
        h_m_to_px[dr] = 0.5 * (np.log1p(ratio) - \
                               np.log1p(-ratio))[:, np.newaxis]
    mask = np.abs(h_m_to_px) > big_field/2
    h_m_to_px[mask] = np.sign(h_m_to_px[mask]) * big_field
    return h_m_to_px




