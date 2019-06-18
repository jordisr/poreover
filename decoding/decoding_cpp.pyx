# distutils: language = c++
# cython: infer_types=True, language_level=3

cimport cython
cimport numpy as np
import numpy as np
from cpython cimport array
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libcpp.string cimport string

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "BeamSearch.h":
    string beam_search(double**, int, string, int, bool)
    string beam_search(double**, double**, int, int, string, int**, int, bool)
    string beam_search(double**, double**, int, int, string, int, bool)
    double forward(double**, int, string, string, bool)

cdef extern from "Gamma.h":
    double pair_gamma_log_envelope(double**, double**, int**, int, int, int)

cdef extern from "PairPrefixSearch.cpp":
    pass

cdef extern from "PairPrefixSearch.h":
    string pair_prefix_search_log(double**, double**, int**, int, int, string)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_forward(y_, label_, alphabet_="ACGT", flipflop=False):

    cdef int U = y_.shape[0]
    cdef int alphabet_size = y_.shape[1]
    cdef string label = label_.encode("UTF-8")
    cdef string alphabet = alphabet_.encode("UTF-8")
    cdef double result

    # Make sure the array a has the correct memory layout (here C-order)
    cdef np.ndarray[double,ndim=2,mode="c"] y = np.asarray(y_, dtype=DTYPE, order="C")

    # Create our helper array
    cdef double** point_to_y = <double **>malloc(U * sizeof(double*))
    try:
        for u in range(U):
            point_to_y[u] = &y[u, 0]
        result = forward(&point_to_y[0], U, label, alphabet, flipflop)
        return(result)
    finally:
        free(point_to_y)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_beam_search(y_, beam_width_=25, alphabet_="ACGT", flipflop=False):

    cdef int U = y_.shape[0]
    cdef int alphabet_size = y_.shape[1]
    cdef string alphabet = alphabet_.encode("UTF-8")
    cdef int beam_width = beam_width_

    # Make sure the array a has the correct memory layout (here C-order)
    cdef np.ndarray[double,ndim=2,mode="c"] y = np.asarray(y_, dtype=DTYPE, order="C")

    # Create our helper array
    cdef double** point_to_y = <double **>malloc(U * sizeof(double*))
    try:
        for u in range(U):
            point_to_y[u] = &y[u, 0]
        decoded_sequence = beam_search(&point_to_y[0], U, alphabet, beam_width, flipflop)
        return(decoded_sequence.decode("UTF-8").lstrip('\x00'))
    finally:
        free(point_to_y)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_beam_search_2d(y1_, y2_, envelope_ranges_=None, beam_width_=25, alphabet_="ACGT", flipflop=False):

    cdef int U = y1_.shape[0]
    cdef int V = y2_.shape[0]
    cdef int alphabet_size = y1_.shape[1]
    cdef int beam_width = beam_width_
    cdef string alphabet = alphabet_.encode("UTF-8")

    # Make sure the array a has the correct memory layout (here C-order)
    cdef np.ndarray[double,ndim=2,mode="c"] y1 = np.asarray(y1_, dtype=DTYPE, order="C")
    cdef np.ndarray[double,ndim=2,mode="c"] y2 = np.asarray(y2_, dtype=DTYPE, order="C")
    cdef np.ndarray[int,ndim=2,mode="c"] envelope_ranges
    if envelope_ranges_ is not None:
        envelope_ranges = np.asarray(envelope_ranges_, dtype=np.intc, order="C")

    # Create our helper array
    cdef double** point_to_y1 = <double **>malloc(U * sizeof(double*))
    cdef double** point_to_y2 = <double **>malloc(V * sizeof(double*))
    cdef int** point_to_envelope_ranges
    if envelope_ranges_ is not None:
        point_to_envelope_ranges = <int **>malloc((U+1) * sizeof(int*))
    try:
        for u in range(U):
            point_to_y1[u] = &y1[u, 0]
        for v in range(V):
            point_to_y2[v] = &y2[v, 0]
        if envelope_ranges_ is not None:
            for u in range(U+1):
                point_to_envelope_ranges[u] = &envelope_ranges[u,0]
            decoded_sequence = beam_search(&point_to_y1[0], &point_to_y2[0], U, V, alphabet, &point_to_envelope_ranges[0], beam_width, flipflop)
        else:
            decoded_sequence = beam_search(&point_to_y1[0], &point_to_y2[0], U, V, alphabet, beam_width, flipflop)
        return(decoded_sequence.decode("UTF-8").lstrip('\x00'))
    finally:
        free(point_to_y1)
        free(point_to_y2)
        if envelope_ranges_ is not None:
            free(point_to_envelope_ranges)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_pair_prefix_search_log(y1_, y2_, envelope_ranges_, alphabet_=b'ACGT'):
#string pair_prefix_search_log(double **y1, double **y2, int **envelope_ranges, int U, int V, string alphabet)

    cdef int U = y1_.shape[0]
    cdef int V = y2_.shape[0]
    cdef int alphabet_size = y1_.shape[1]
    cdef string alphabet = alphabet_.encode("UTF-8")

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
        decoded_sequence = pair_prefix_search_log(&point_to_y1[0], &point_to_y2[0], &point_to_envelope_ranges[0], U, V, alphabet)
        return(decoded_sequence)
    finally:
        free(point_to_y1)
        free(point_to_y2)
        free(point_to_envelope_ranges)

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
