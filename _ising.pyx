cimport cython
import numpy as np
cimport numpy as np


#DTYPE = np.float
#ctypedef np.double_t DTYPE_t

min_inf = -10000
max_inf = 500

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
    logZp = np.maximum(logZp, min_inf)
    # when maxinf is too small this leads to problems at the boundaries
    logZp = np.minimum(logZp, max_inf)
    return np.exp(logZp)

