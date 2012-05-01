import numpy as np
from scipy import ndimage, stats
import math
from math import exp, log1p
from scipy import stats

min_inf = -10000

#----------------Microconical Ising chain -----------------------------

def log_exp_plus_exp(a, b):
    """
    this is not used yet
    """
    #mininf = - 10000
    if a > b:
        return a + log1p(exp(b - a))
    else:
        return b + log1p(exp(a - b))


def _build_logZp(h, J):
    """
    Build partial left partition function
    Index convention: 0 for -1, 1 for +1 spin
    """
    N = len(h)
    logZp = min_inf * np.ones((N, 2*N + 3, 2))
    logZp[0, N, 0] = -h[0]
    logZp[0, N + 2, 1] = h[0]
    for n in range(1, N):
        for v in range(N - n, N + n + 3, 2):
            logZp[n, v, 0] = log_exp_plus_exp(-h[n] + J[n-1] + 
              logZp[n - 1, v + 1, 0], -h[n] - J[n-1] + logZp[n - 1, v + 1, 1])
            logZp[n, v, 1] = log_exp_plus_exp(h[n] - J[n-1] + 
              logZp[n - 1, v - 1, 0], h[n] + J[n-1] + logZp[n - 1, v - 1, 1])
    logZp = logZp[:, 1:-1, :]
    # This is very important -- apparently
    logZp = np.maximum(logZp, min_inf)
    #logZp = np.minimum(logZp, )
    return np.exp(logZp)


def _build_left_right(h, J):
    """
    Build two partial partition functions, left and right.

    For computing the right partition function Tp, we just reverse the order
    of h and J, and compute the left partition function, and then reverse its
    order.
    """
    Zp = _build_logZp(h, J)
    Tp = _build_logZp(h[::-1], J[::-1])
    Tp = Tp[::-1]
    return Zp, Tp

def gaussian_weight(s, s0, beta=1.):
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

def solve_microcanonical_chain(h, J, s0, error=2):
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
    prob = np.zeros((2, N))
    Zp, Tp = _build_left_right(h, J)
    # indices for the sum of spins
    u = np.arange(2 * N + 1)
    u = u[:, None, None, None]
    # v = 
    err = np.arange(-error, error + 1)[None, :, None, None]
    Us = u - N
    v = (s0 + 2*N - u + err).astype(np.int)
    Vs = v - N
    Vs[np.logical_or(v < 0, v >= 2*N + 1)] = 1.e5 # hack, we don't want this
    # to happen
    v[np.logical_or(v < 0, v >= 2*N + 1)] = 0
    s_m = np.arange(2)[None, None, :, None]
    S_m = 2*(s_m - 0.5)
    s_p = np.arange(2)[None, None, None, :]
    S_p = 2*(s_p - 0.5)
    # Now we write the probability of spin i, which is given by
    # proba[s_i] = sum_{u, v, s_{n-1}, s_{n+1}}
    #         Z_{i-1}(u,s_{n-1}) T_{n+1}(v,s_{n+1}) x
    #        exp[J(s_{n-1}s_n+s_{n+1}s_n] exp[h_n s_n] w(s_n+u+v)
    for si in range(1, N-1):
        s_n = -1
        # sum over u, v, s_m and s_p with broadcasting
        prob[0, si] =  (Zp[si - 1, u, s_m] * Tp[si + 1, v, s_p] * \
                np.exp(J[si-1] * s_n * S_m + J[si] * s_n * S_p +
                h[si] * s_n) * gaussian_weight(s_n + Us + Vs, s0)).sum()
        s_n = 1
        prob[1, si] =  (Zp[si - 1, u, s_m] * Tp[si + 1, v, s_p] * \
            np.exp(J[si-1] * s_n * S_m + J[si] * s_n * S_p +
                h[si] * s_n) * gaussian_weight(s_n + Us + Vs, s0)).sum()
    u, Us = u.ravel(), Us.ravel()
    # Manage boundaries
    prob[0, 0] = (Tp[0, u, 0] * gaussian_weight(Us, s0)).sum()
    prob[1, 0] = (Tp[0, u, 1] * gaussian_weight(Us, s0)).sum()
    prob[0, -1] = (Zp[-1, u, 0] * gaussian_weight(Us, s0)).sum()
    prob[1, -1] = (Zp[-1, u, 1] * gaussian_weight(Us, s0)).sum()
    return prob

def solve_microcanonical_h(h, J, s0, error=1):
    """
    Compute local magnetization for microcanonical Ising chain
    """
    proba = solve_microcanonical_chain(h, J, s0, error=error)
    ratio = (proba[1] - proba[0])/(proba[1] + proba[0])
    ratio = np.maximum(-1 + 1.e-16, ratio)
    ratio = np.minimum(1 - 1.e-16, ratio)
    res = np.arctanh(ratio)
    res[np.isnan(res)] = 0
    res[np.isinf(res)] = 0
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




