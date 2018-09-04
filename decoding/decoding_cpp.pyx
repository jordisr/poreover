# cython: infer_types=True
cimport cython
cimport numpy as np
import numpy as np
from cpython cimport array
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "Gamma.cpp":
    pass

cdef extern from "Gamma.h":
    #double pair_gamma_log_envelope(double[][3], double[][3], int[][2], int, int)
    #double pair_gamma_log_envelope(double y1[][3], double y2[][3], int envelope_ranges[][2], int U, int V)
    #double pair_gamma_log_envelope(double[:,:], double[:,:], int[:,:], int, int)
    #double pair_gamma_log_envelope(double*, double*, int*, int, int)
    double pair_gamma_log_envelope(double**, double**, int**, int, int, int)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_pair_gamma_log_envelope(y1_, y2_, envelope_ranges_):

    cdef int U = y1_.shape[0]
    cdef int V = y2_.shape[0]
    cdef int alphabet_size = y1_.shape[1]

    # Make sure the array a has the correct memory layout (here C-order)
    cdef np.ndarray[double,ndim=2,mode="c"] y1 = np.asarray(y1_, dtype=DTYPE, order="C")
    cdef np.ndarray[double,ndim=2,mode="c"] y2 = np.asarray(y2_, dtype=DTYPE, order="C")
    cdef np.ndarray[int,ndim=2,mode="c"] envelope_ranges = np.asarray(envelope_ranges_, dtype=np.intc, order="C")

    # Create our helper array
    cdef double** point_to_y1 = <double **>malloc(U * sizeof(double*))
    cdef double** point_to_y2 = <double **>malloc(V * sizeof(double*))
    cdef int** point_to_envelope_ranges = <int **>malloc((U+1) * sizeof(int*))
    try:
        for u in range(U):
            point_to_y1[u] = &y1[u, 0]
        for v in range(V):
            point_to_y2[v] = &y2[v, 0]
        for u in range(U+1):
            point_to_envelope_ranges[u] = &envelope_ranges[u,0]
        # Call the C function that expects a double**
        testy_mctest = pair_gamma_log_envelope(&point_to_y1[0], &point_to_y2[0], &point_to_envelope_ranges[0], U, V, alphabet_size)
        print(testy_mctest)
    finally:
        free(point_to_y1)
        free(point_to_y2)
        free(point_to_envelope_ranges)

    #y1 = np.ascontiguousarray(y1)
    #y2 = np.ascontiguousarray(y2)
    #cdef double[:,:,::1] y1_view = y1
    #cdef double[:,:,::1] y2_view = y2

    #testy_mctest = pair_gamma_log_envelope(y1_view, y2_view, envelope_ranges, u, v)
    #testy_mctest = pair_gamma_log_envelope(&y1[0,0], &y2[0,0], &envelope_ranges[0,0], u, v)

    #print(testy_mctest)

y1 = np.array([[-0.223144, -2.30259, -2.30259],[-2.30259, -1.20397, -0.510826],[-0.356675, -1.60944, -2.30259],[-2.30259, -2.30259, -0.223144]]).astype(DTYPE)
y2 = np.array([[-0.356675,-1.60944,-2.30259],[-1.60944,-1.20397,-0.693147],[-0.356675,-1.60944,-2.30259],[-2.99573,-2.99573,-0.105361]]).astype(DTYPE)
envelope_ranges = np.array([[0,4],[0,4],[0,4],[0,4],[0,4]]).astype(np.intc)
cpp_pair_gamma_log_envelope(y1, y2, envelope_ranges)
