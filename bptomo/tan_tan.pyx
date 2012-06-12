import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "tan_tan.h":
    float atanh_th_th(float , float )

cdef fast_atanh_th_th(float x, float y):
    return atanh_th_th(x, y)

cimport cython
@cython.boundscheck(False)
def fast_mag_chain(np.ndarray[DTYPE_t, ndim=1] h not None, float J, float hext):
    
    cdef int N = len(h)
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros((N), dtype=DTYPE)
    cdef int i
    cdef float magtot
    cdef np.ndarray[DTYPE_t, ndim=1] hloc = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] mag = np.zeros((N), dtype=DTYPE)
    for i in range(1, N):
        u[i] = fast_atanh_th_th(J, hext + u[i - 1] + h[i - 1])
    for i in range(N - 2, 0, -1):
        v[i] = fast_atanh_th_th(J, hext + v[i + 1] + h[i + 1])
    hloc = u + v + h + hext
    mag = np.tanh(hloc)
    magtot = mag.sum()
    #TODO big_field threshold?
    return magtot, hloc

@cython.boundscheck(False)
def fast_mag_chain_nu(np.ndarray[DTYPE_t, ndim=1] h not None, np.ndarray[DTYPE_t, ndim=1] J not None, float hext):
    
    cdef int N = len(h)
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros((N), dtype=DTYPE)
    cdef int i
    cdef float magtot
    cdef np.ndarray[DTYPE_t, ndim=1] hloc = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] mag = np.zeros((N), dtype=DTYPE)
    for i in range(1, N):
        u[i] = fast_atanh_th_th(J[i-1], hext + u[i - 1] + h[i - 1])
    for i in range(N - 2, 0, -1):
        v[i] = fast_atanh_th_th(J[i], hext + v[i + 1] + h[i + 1])
    hloc = u + v + h + hext
    mag = np.tanh(hloc)
    magtot = mag.sum()
    return magtot, hloc

