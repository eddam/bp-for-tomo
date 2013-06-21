import numpy as np
from tan_tan import fast_mag_chain_nu, derivative_passing, \
                    mag_chain_uncoupled, mag_chain_uncoupled_derivative, \
                    fast_mag_chain_uncoupled
# from solve import solve_canonical_h as solve_canonical_h_cython
from scipy import optimize
from math import atanh

#-------------------------- Canonical formulation ----------------

def mag_chain_uncoupled_error(hext, hi, y):
    return mag_chain_uncoupled(hi, hext) - y

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

def mag_chain_for_opt(hext, h, J, y, full_output):
    return mag_chain(h, J, hext, full_output) - y

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
    epsilon = .05
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
    iter_nb = 1
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



def solve_canonical_h_opt(h, J, y, hext=None):
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
    epsilon = .1
    N = float(len(h))
    y = min(y, N)
    y = max(y, -N)
    hext = None
    if hext is not None:
        hext = optimize.newton(mag_chain_for_opt, hext,
                    args=(h, J, y, False), tol=epsilon / N)
        mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
    else:
        """
        hext = 0
        mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
        if abs(mag_tot - y) < epsilon:
            return hloc, hext
        if mag_tot < y:
            hmin = 0
            hext = 8
            while y - mag_chain(h, J, hext) > 0:
                hmin = hext
                hext *= 2
            hmax = hext
        else:
            hmax = 0
            hext = -8
            while mag_chain(h, J, hext) - y > 0:
                hmax = hext
                hext *= 2
            hmin = hext
        """
        try:
            hext, res = optimize.brentq(mag_chain_for_opt, -6, 6,
                        args=(h, J, y, False), xtol=epsilon / N, 
                        full_output=True)
            print res.iterations
        except ValueError:
            hext = optimize.brentq(mag_chain_for_opt, -20, 20,
                        args=(h, J, y, False), xtol=epsilon / N)
        mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
    return hloc, hext


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
            mag_tot = mag_chain(h, J, hext)
            while y - mag_tot > epsilon:
                hmin = hext
                hext *= 2
                mag_tot = mag_chain(h, J, hext)
            hmax = hext
        else:
            hmax = hext
            hext = min(-8, hext - 1)
            mag_tot = mag_chain(h, J, hext)
            while mag_tot - y > epsilon:
                hmax = hext
                hext *= 2
                mag_tot = mag_chain(h, J, hext)
            hmin = hext
        # dichotomy
        while (abs(mag_tot - y) > epsilon and
              (abs(mag_tot - y) > dmax or
               np.abs(hext) > 6.5) and iter_nb < 200):
            iter_nb += 1
            hext = 0.5 * (hmin + hmax)
            mag_tot, hloc = mag_chain(h, J, hext, full_output=True)
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


# ---------------------------------------------------------------

min_inf = -10000
max_inf = 500

# ------------------ Solving Ising model for one projection -----------
def solve_line(field, Js, y, big_field=400, hext=None):
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

    Returns
    -------

    hloc: 1-d ndarray, same shape as field
        local magnetization
    """
    # Handle large fields
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if np.all(mask_blocked) and np.abs(np.sign(field).sum() - y) < 0.1:
        return 0.5 * field, hext
    # Solve Ising chain
    hloc, hext = solve_canonical_h(field, Js, y, hext)
    mask_blocked = np.abs(hloc) > big_field
    hloc[mask_blocked] = big_field * np.sign(hloc[mask_blocked])
    # Remove initial field
    hloc -= field
    return hloc, hext

def solve_uncoupled_line(field, y, big_field=400, hext=None):
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

    Returns
    -------

    hloc: 1-d ndarray, same shape as field
        local magnetization
    """
    if np.abs(y) >= len(field) - 0.5:
        return np.sign(y) * 10
    # Handle large fields
    mask_blocked = np.abs(field) > big_field
    field[mask_blocked] = big_field * np.sign(field[mask_blocked])
    if np.all(mask_blocked) and np.abs(np.sign(field).sum() - y) < 0.1:
        return 0
    if hext is None:
        hext = 0
    # Solve Ising chain
    hext = my_bisect(field, y, hext, tol=0.2)
    return hext

def my_bisect(field, y, hext, tol=0.1):
    niter_max = 30
    l = float(len(field))
    err_init =  mag_chain_uncoupled_error(hext, field, y) 
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
        err = mag_chain_uncoupled_error(xm, field, y)
        if abs(err) < tol:
            return xm
        if err <= 0:
            vmin = xm
        dx *= 0.5
        i += 1
    if i == niter_max:
        raise ValueError
