import numpy as np
from scipy import ndimage, stats
import math
from math import exp, log1p
from scipy import stats
from _ising import _build_logZp, log_exp_plus_exp, solve_microcanonical_chain_pyx, solve_microcanonical_chain
from tan_tan import fast_mag_chain_nu, derivative_passing

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

def mag_chain_deriv(h, J, hext):
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
    magtot, deriv, hloc = derivative_passing(h, J, hext)
    return magtot, deriv, hloc


#@profile
def solve_canonical_h(h, J, y, hext=None):
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
    epsilon= .05
    if hext is None or np.isnan(hext):
        hext = 0
    hext = 0
    N = float(len(h))
    y = min(y, N)
    y = max(y, -N)
    mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
    if abs(mag_tot - y) < epsilon:
        return hloc, hext
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
    while abs(mag_tot - y) > epsilon and iter_nb < 200:
        iter_nb += 1
        hext = 0.5 * (hmin + hmax)
        mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
        if mag_tot < y:
            hmin = hext
        else:
            hmax = hext
    return hloc, hext

def solve_canonical_h_newton(h, J, y, hext_init=None):
    """
    Solve Ising chain in the canonical formulation.

    Parameters
    ----------

    h: 1-d ndarray of floats
        local field

    J: 1-d ndarray of floats
        coupling between spins

    y: total magnetization

    Returns
    -------

    hloc: 1-d ndarray of floats
        local magnetization
    """
    if hext_init is None:
        hext_init = np.sign(y)
    epsilon= .05
    N = len(h)
    y = min(y, N)
    y = max(y, -N)
    hext = hext_init
    mag_tot, deriv_mag_tot, hloc = mag_chain_deriv(h, J, hext)
    iter_nb = 1
    dmax = 1.5
    if np.abs(mag_tot - y) > dmax or deriv_mag_tot < 1.e-5:
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
        # dichotomy
        while abs(mag_tot - y) > dmax and iter_nb < 200:
            iter_nb += 1
            hext = 0.5 * (hmin + hmax)
            mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
            if mag_tot < y:
                hmin = hext
            else:
                hmax = hext
    iter_first = iter_nb
    hext_first = hext
    mag_tot_first = mag_tot
    iter_nb = 1
    while abs(mag_tot - y) > epsilon and iter_nb < 30:
        mag_tot, deriv_mag_tot, hloc = mag_chain_deriv(h, J, hext)
        hext -= (mag_tot - y) / deriv_mag_tot
        iter_nb += 1
    if iter_nb >= 30:
        if hmin is None:
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
        # dichotomy
        while abs(mag_tot - y) > epsilon and iter_nb < 200:
            iter_nb += 1
            hext = 0.5 * (hmin + hmax)
            mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
            if mag_tot < y:
                hmin = hext
            else:
                hmax = hext
    if iter_nb > 200:
        print "failed"
    #print iter_nb
    if np.isnan(hext):
        raise ValueError
    return hloc, hext



# ---------------------------------------------------------------

min_inf = -10000
max_inf = 500

#----------------Microconical Ising chain -----------------------------



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

def solve_line(field, Js, y, onsager=1, big_field=400, use_micro=False,
                                hext=None):
    """
    Solve Ising chain

    Parameters
    ----------

    field: 1-d ndarray
        Local field on the spins

    Js: float
        Coupling between spins

    y: float
        Sum of the spins (value of the projection)

    onsager: float

    big_field: float

    use_micro: bool, default False

    Returns
    -------

    hloc: 1-d ndarray, same shape as field
        local magnetization
    """
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if np.all(mask_blocked) and np.abs(np.sign(field).sum() - y) < 0.1:
        return (1.5 - onsager) * field, hext
    elif use_micro is False or np.sum(~mask_blocked) > 25:
        #hloc, hext = solve_canonical_h(field, Js, y, hext)
        hloc, hext = solve_canonical_h_newton(field, Js, y, hext)
        mask_blocked = np.abs(hloc) > big_field
        hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
        hloc -= onsager * field
        return hloc, hext
    else:
        hloc = solve_microcanonical_h(field, Js, y)
        mask_blocked = np.abs(hloc) > big_field
        hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
        hloc -= onsager * field
        return hloc, hext


