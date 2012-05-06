cimport cython
import numpy as np
cimport numpy as np


#DTYPE = np.float
#ctypedef np.double_t DTYPE_t

cdef double min_inf = -10000
cdef double max_inf = 500

cdef extern from "math.h":
    double exp(double)
    double log1p(double)


def log_exp_plus_exp(float a, float b):
    if a > b:
        return a + log1p(exp(b - a))
    else:
        return b + log1p(exp(a - b))

@cython.boundscheck(False)
def _build_logZp(np.ndarray[dtype=np.double_t, ndim=1] h not None,
                     np.ndarray[dtype=np.double_t, ndim=1] J not None):
    """
    Build partial left partition function
    Index convention: 0 for -1, 1 for +1 spin
    """
    cdef int N = len(h)
    cdef np.ndarray[dtype=np.double_t, ndim=3] logZp = \
                            min_inf * np.ones((N, 2*N + 3, 2))
    logZp[0, N, 0] = -h[0]
    logZp[0, N + 2, 1] = h[0]
    cdef int n, v
    for n in range(1, N):
        for v in range(N - n, N + n + 3, 2):
            logZp[n, v, 0] = - h[n] + log_exp_plus_exp(J[n-1] +
              logZp[n - 1, v + 1, 0], - J[n-1] + logZp[n - 1, v + 1, 1])
            logZp[n, v, 1] = h[n] + log_exp_plus_exp(- J[n-1] +
              logZp[n - 1, v - 1, 0], J[n-1] + logZp[n - 1, v - 1, 1])
    logZp = logZp[:, 1:-1, :]
    # This is very important -- apparently
    logZp = np.minimum(logZp, max_inf)
    cdef float m = np.max(logZp)
    if m > 200:
        logZp -= m/2.
    # when maxinf is too small this leads to problems at the boundaries
    logZp = np.maximum(logZp, min_inf)
    return (logZp)

def _build_left_right(np.ndarray[dtype=np.double_t, ndim=1] h,
                      np.ndarray[dtype=np.double_t, ndim=1] J):
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


def log_gaussian_weight(s, s0, beta=1.):
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

@cython.boundscheck(False)
def solve_microcanonical_chain_broad(np.ndarray[dtype=np.double_t, ndim=1]
                                h not None,
                                np.ndarray[dtype=np.double_t, ndim=1]
                                J not None,
                                float s0):
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
    cdef int N = len(h)
    cdef np.ndarray[dtype=np.double_t, ndim=2] prob = np.zeros((2, N))
    cdef np.ndarray[dtype=np.double_t, ndim=3] Zp, Tp
    Zp, Tp = _build_left_right(h, J)
    #print Zp.min(), Tp.min(), Zp.max(), Tp.max()
    Zp = np.exp(Zp)
    Tp = np.exp(Tp)
    # indices for the sum of spins
    #u = np.arange(2 * N + 1)
    cdef np.ndarray[dtype=np.int_t, ndim=2] u = \
                    np.arange(2 * N + 1)[:, np.newaxis]
    cdef int error = 1
    cdef np.ndarray[dtype=np.int_t, ndim=2] err = \
                    np.arange(-error, error + 1)[None, :]
    cdef np.ndarray[dtype=np.int_t, ndim=2] v
    cdef np.ndarray[dtype=np.int_t, ndim=2] Vs
    v = (s0 + 2*N - u + err).astype(np.int)
    v = np.arange(s0 + 2*N -1, s0 + 2*N + 2, 
                2).astype(np.int)[np.newaxis, :] - u
    Vs = v - N
    Vs[np.logical_or(v < 0, v >= 2*N + 1)] = 1.e5 # gaussian_weight excludes  
    v[np.logical_or(v < 0, v >= 2*N + 1)] = 0
    # Now we write the probability of spin i
    cdef int si, s_n
    for si in range(0, N):
        s_n = -1
        prob[0, si] = (Zp[si, u, 0] * Tp[si, v, 0] *\
                exp(- h[si] * s_n) * \
                gaussian_weight(- s_n + u + Vs - N, s0)).sum()
        s_n = 1
        prob[1, si] = (Zp[si, u, 1] * Tp[si, v, 1] *\
                    exp(- h[si] * s_n) * \
                    gaussian_weight(- s_n + u + Vs - N, s0)).sum()
    mask = prob == 0
    prob[mask] = 1
    res = np.log(prob)
    res[mask] = min_inf
    return res


@cython.boundscheck(False)
def solve_microcanonical_chain_pyx(np.ndarray[dtype=np.double_t, ndim=1]
                                h not None,
                                np.ndarray[dtype=np.double_t, ndim=1]
                                J not None,
                                float s0):
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
    cdef int N = len(h)
    cdef np.ndarray[dtype=np.double_t, ndim=2] prob = min_inf * np.ones((2, N))
    cdef np.ndarray[dtype=np.double_t, ndim=3] Zp, Tp
    Zp, Tp = _build_left_right(h, J)
    # indices for the sum of spins
    #u = np.arange(2 * N + 1)
    cdef np.ndarray[dtype=np.int_t, ndim=1] u = np.arange(2 * N + 1)
    cdef np.ndarray[dtype=np.int_t, ndim=1] v
    # Now we write the probability of spin i
    cdef int si, uu, vv, s_n
    for si in range(0, N):
        for uu in u:
            v = np.arange(s0 + 2*N - uu -1, s0 + 2*N -uu + 2, 2).astype(np.int)
            v = v[np.logical_and(v >= 0, v < 2*N + 1)]
            for vv in v:
                s_n = -1
                prob[0, si] = log_exp_plus_exp(prob[0, si],
                    Zp[si, uu, 0] + Tp[si, vv, 0] \
                    - h[si] * s_n  \
                    + log_gaussian_weight(- s_n + uu + vv - 2*N, s0))
                s_n = 1
                prob[1, si] = log_exp_plus_exp(prob[1, si],
                    Zp[si, uu, 1] + Tp[si, vv, 1]  \
                    - h[si] * s_n  \
                    + log_gaussian_weight(- s_n + uu + vv - 2*N, s0))
    return np.minimum(prob, max_inf)

