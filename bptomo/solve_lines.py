import numpy as np
from _solve_lines import fast_mag_chain_nu, derivative_passing, \
                    mag_chain_uncoupled, mag_chain_uncoupled_derivative, \
                    fast_mag_chain_uncoupled, cython_mag_mag_error, cython_mag
from scipy import optimize
from math import atanh, tanh

# -------Coupled lines: solving Ising model for one projection -----------


def solve_line(field, Js, y, big_field=15, hext=None):
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

    big_field: float

    hext: float
        Lagrange multiplier in order to impose the sum of the magnetizations

    Returns
    -------

    hloc: 1-d ndarray, same shape as field
        local magnetization (with the Onsager term removed)
    """
    # Handle large fields
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if np.all(mask_blocked) and np.abs(np.sign(field).sum() - y) < 0.1:
        return 0.5 * field, hext
    # Solve Ising chain
    hloc, hext = _solve_canonical_h(field, Js, y, hext)
    mask_blocked = np.abs(hloc) > big_field
    hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
    # Remove initial field
    hloc -= field
    return hloc, hext


def _solve_canonical_h(h, J, y, hext=None):
    """
    Solve Ising chain in the canonical formulation.

    Parameters
    ----------

    h: 1-d ndarray of floats
        local field

    y: total magnetization

    J: 1-d ndarray of floats
        coupling between spins

    hext: float
        Lagrange multiplier in order to impose the sum of the magnetizations

    Returns
    -------

    hloc: 1-d ndarray of floats
        local magnetization
    """
    epsilon = .05
    if hext is None or np.isnan(hext):
        hext = 0
    hext = 0
    N = float(len(h))
    y = min(y, N)
    y = max(y, -N)
    mag_tot, hloc = _mag_chain(h, J, hext, full_output=True)
    if abs(mag_tot - y) < epsilon:
        return hloc, hext
    if mag_tot < y:
        hmin = 0
        hext = 8
        while y - _mag_chain(h, J, hext) > epsilon:
            hmin = hext
            hext *= 2
        hmax = hext
    else:
        hmax = 0
        hext = -8
        while _mag_chain(h, J, hext) - y > epsilon:
            hmax = hext
            hext *= 2
        hmin = hext
    mag_tot = 2 * N
    iter_nb = 1
    # dichotomy
    while abs(mag_tot - y) > epsilon and iter_nb < 200:
        iter_nb += 1
        hext = 0.5 * (hmin + hmax)
        mag_tot, hloc = _mag_chain(h, J, hext, full_output=True)
        if mag_tot < y:
            hmin = hext
        else:
            hmax = hext
    return hloc, hext


def _mag_chain(h, J, hext, full_output=False):
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


# ------------------- Uncoupled lines (no interaction between neighbors)------

def solve_uncoupled_line(field, y, big_field=15, hext=None):
    """
    Solve Ising chain

    Parameters
    ----------

    field: 1-d ndarray
        Local field on the spins

    y: float
        Sum of the spins (value of the projection)

    big_field: float
        Largest possible value of the external field, in order to
        avoid overflow problems. It's not clear yet whether it should be
        large or not.

    Returns
    -------

    hext: float
        External field (Lagrange multiplier)

    Notes
    -----

    The returned value hext solves

        sum_i tanh(field[i] + hext) = y

    For now a bisection method is used, with some care taken to avoid
    overflow.

    TODO: use the previous value of hext, in some cases it might be possible
    to use newton's methods to converge faster.
    """
    if abs(y) >= len(field) - 0.5:
        return np.sign(y) * big_field
    # Handle large fields
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if hext is None:
        hext = 0
    # Solve Ising chain
    hext = _bisect_mag(field, y, big_field)
    #hext = bisect_field(field, y, hext, tol=0.2)
    return hext


def _bisect_mag(field, y, big_field):
    mag = np.tanh(field)
    magsup = tanh(big_field)
    try:
        Mext = optimize.brentq(cython_mag_mag_error, -magsup, magsup,
                            xtol=1.e-2, args=(mag, y))
    except ValueError:
        if cython_mag_mag_error(magsup, mag, y) > 0:
            return big_field
        elif cython_mag_mag_error(- magsup, mag, y) < 0:
            return -big_field
    return atanh(Mext)


def _mag_chain_uncoupled_error(hext, hi, y):
    return mag_chain_uncoupled(hi, hext) - y


def _bisect_field(field, y, hext, tol=0.1):
    niter_max = 30
    l = float(len(field))
    err_init =  _mag_chain_uncoupled_error(hext, field, y) 
    dx = 10
    if err_init < 0:
        vmin = hext
        vmax = - field.min() + atanh(y/l) + 0.1
        dx = 0.5 * (vmax - vmin)
    else:
        vmin = - field.max() + atanh(y/l) - 0.1
        vmax = hext
        dx = 0.5 * (vmax - vmin)
    i = 0
    while i < niter_max:
        xm = vmin + dx
        err = _mag_chain_uncoupled_error(xm, field, y)
        if abs(err) < tol:
            return xm
        if err <= 0:
            vmin = xm
        dx *= 0.5
        i += 1
    if i == niter_max:
        raise ValueError


# ---------------------- Unused functions (to be removed?) ---------------


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



def solve_canonical_h_newton(h, J, y, hext_init=None):
    """
    Solve Ising chain in the canonical formulation.

    Use Newton's method when the initial guess is close enough.

    Parameters
    ----------

    h: 1-d ndarray of floats
        local field

    J: 1-d ndarray of floats
        coupling between spins

    y: float
        total magnetization

    hext_init: float
        initial guess

    Returns
    -------

    hloc: 1-d ndarray of floats
        local magnetization
    """
    if hext_init is None:
        hext_init = np.sign(y)
    epsilon = .05
    N = len(h)
    y = min(y, N)
    y = max(y, -N)
    hext = hext_init
    mag_tot, deriv_mag_tot, hloc = mag_chain_deriv(h, J, hext)
    iter_nb = 1
    dicho = False
    dmax = 1  # heuristic, 1 or 1.5?
    # First, use the dichotomy if we are too far from the expected result
    if np.abs(mag_tot - y) > dmax or deriv_mag_tot < 1.e-5 or abs(hext) > 6.5:
        dicho = True
        # Search bounds for the dichotomy
        if mag_tot < y:
            hmin = hext
            hext = max(8, hext + 1)
            mag_tot = _mag_chain(h, J, hext)
            while y - mag_tot > epsilon:
                hmin = hext
                hext *= 2
                mag_tot = _mag_chain(h, J, hext)
            hmax = hext
        else:
            hmax = hext
            hext = min(-8, hext - 1)
            mag_tot = _mag_chain(h, J, hext)
            while mag_tot - y > epsilon:
                hmax = hext
                hext *= 2
                mag_tot = _mag_chain(h, J, hext)
            hmin = hext
        # dichotomy
        while (abs(mag_tot - y) > epsilon and
              (abs(mag_tot - y) > dmax or
               np.abs(hext) > 6.5) and iter_nb < 200):
            iter_nb += 1
            hext = 0.5 * (hmin + hmax)
            mag_tot, hloc = _mag_chain(h, J, hext, full_output=True)
            if mag_tot < y:
                hmin = hext
            else:
                hmax = hext
    iter_first = iter_nb
    if iter_first > 20:
        print "iter too much"
    iter_nb = 0
    # Now that we are close enough, we use Newton's method
    if dicho is False and deriv_mag_tot > 0:
        hext -= (mag_tot - y) / deriv_mag_tot
    while abs(mag_tot - y) > epsilon and iter_nb < 40:
        mag_tot, deriv_mag_tot, hloc = mag_chain_deriv(h, J, hext)
        if deriv_mag_tot == 0:
            raise ValueError
        hext -= (mag_tot - y) / deriv_mag_tot
        iter_nb += 1
    return hloc, hext



