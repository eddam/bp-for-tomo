import numpy as np
cimport numpy as np
from libc.math cimport tanh, atanh 

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "tan_tan.h":
    float atanh_th_th(float , float )

cdef float fast_atanh_th_th(float x, float y):
    return atanh_th_th(x, y)

cdef float my_atanh_th_th(float x, float y):
    return atanh(tanh(x) * tanh(y))

cdef extern from "tan_tan.h":
    float deriv_atanh_th_th(float , float , float )

cdef deriv_fast_atanh_th_th(float x, float y, float z):
    return deriv_atanh_th_th(x, y, z)


cimport cython

@cython.boundscheck(False)
def cython_mag_mag_error(float Mext, np.ndarray[DTYPE_t, ndim=1] mag not None,
                float y):
    cdef float magtot=0
    cdef int i
    cdef unsigned int N = len(mag)
    for i in range(N):
        magtot += (mag[i] + Mext) / (1. + mag[i] * Mext)
    return y - magtot


@cython.boundscheck(False)
def cython_mag(np.ndarray[DTYPE_t, ndim=1] field not None):
    cdef unsigned int N = len(field)
    cdef np.ndarray[DTYPE_t, ndim=1] mag = np.zeros((N), dtype=DTYPE)
    cdef int i
    for i in range(N):
        mag[i] = tanh(field[i])
    return mag


@cython.boundscheck(False)
def mag_chain_uncoupled(np.ndarray[DTYPE_t, ndim=1] h not None,
                float hext):
    cdef float magtot=0
    cdef int i
    cdef unsigned int N = len(h)
    for i in range(N):
        magtot += tanh(h[i] + hext)
    return magtot


@cython.boundscheck(False)
def mag_chain_uncoupled_derivative(float hext,
        np.ndarray[DTYPE_t, ndim=1] h not None, float y):
    cdef float acc = 0
    cdef int i
    cdef float val_tmp
    cdef unsigned int N = len(h)
    for i in range(N):
        val_tmp = tanh(h[i] + hext)
        acc += 1 - val_tmp ** 2
    return acc


@cython.boundscheck(False)
def fast_mag_chain_uncoupled(np.ndarray[DTYPE_t, ndim=1] h not None,
                float hext):
    cdef float magtot=0
    cdef int i
    cdef float vmin, vmax
    vmin = -5. - hext
    vmax = 5. - hext
    cdef float val_tmp
    cdef unsigned int N = len(h)
    for i in range(N):
        val_tmp = h[i]
        if h[i] < vmin:
            magtot -= 1
        if h[i] > vmax:
            magtot += 1
        else:
            magtot += tanh(h[i] + hext)
    return magtot


@cython.boundscheck(False)
def fast_mag_chain_nu(np.ndarray[DTYPE_t, ndim=1] h not None, np.ndarray[DTYPE_t, ndim=1] J not None, float hext):
    cdef unsigned int N = len(h)
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros(N, dtype=DTYPE)
    cdef int i
    cdef float magtot = 0
    cdef np.ndarray[DTYPE_t, ndim=1] hloc = np.zeros(N, dtype=DTYPE)
    for i in range(1, N):
        u[i] = fast_atanh_th_th(J[i-1], hext + u[i - 1] + h[i - 1])
    for i in range(N - 2, 0, -1):
        v[i] = fast_atanh_th_th(J[i], hext + v[i + 1] + h[i + 1])
    for i in range(N):
        hloc[i] = u[i] + v[i] + h[i] + hext
        magtot += tanh(hloc[i])
    return magtot, hloc

@cython.boundscheck(False)
def simple_ising(np.ndarray[DTYPE_t, ndim=1] h not None, float J):
    cdef unsigned int N = len(h)
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros(N, dtype=DTYPE)
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] hloc = np.zeros(N, dtype=DTYPE)
    for i in range(1, N):
        u[i] = fast_atanh_th_th(J, u[i - 1] + h[i - 1])
    for i in range(N - 2, 0, -1):
        v[i] = fast_atanh_th_th(J, v[i + 1] + h[i + 1])
    for i in range(N):
        hloc[i] = u[i] + v[i]
    return hloc


@cython.boundscheck(False)
def fast_mag_chain_derivative(np.ndarray[DTYPE_t, ndim=1] h not None, np.ndarray[DTYPE_t, ndim=1] J not None, float hext):
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
    return magtot, N - (mag**2).sum(), hloc

@cython.boundscheck(False)
def derivative_passing(np.ndarray[DTYPE_t, ndim=1] h not None, np.ndarray[DTYPE_t, ndim=1] J not None, float hext):
    cdef int N = len(h)
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros((N))
    cdef np.ndarray[DTYPE_t, ndim=1] du = np.zeros((N))
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros((N))
    cdef np.ndarray[DTYPE_t, ndim=1] dv = np.zeros((N))
    cdef float magtot
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] hloc = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dhloc = np.zeros((N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] mag = np.zeros((N), dtype=DTYPE)
    for i in range(1, N):
        htot_tmp = hext + u[i - 1] + h[i - 1]
        u[i] = fast_atanh_th_th(J[i-1], htot_tmp)
        du[i] = deriv_atanh_th_th(J[i - 1], htot_tmp, du[i - 1])
    for i in range(N - 2, 0, -1):
        htot_tmp = hext + v[i + 1] + h[i + 1]
        v[i] = fast_atanh_th_th(J[i], htot_tmp)
        dv[i] = deriv_atanh_th_th(J[i - 1], htot_tmp, dv[i + 1])
    hloc = u + v + h + hext
    dhloc = 1 + du + dv
    mag = np.tanh(hloc)
    magtot = mag.sum()
    return magtot, ((1 - mag**2) * dhloc).sum(), hloc

