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
    string beam_search(double**, int, string, int, string)
    string beam_search(double**, double**, int, int, string, int**, int, string)
    string beam_search(double**, double**, int, int, string, int, string)
    double forward(double**, int, string, string, string)

cdef extern from "Forward.h":
    string viterbi_acceptor_poreover(double**, int, int, string, string alphabet)

cdef extern from "Gamma.h":
    double pair_gamma_log_envelope(double**, double**, int**, int, int, int)

cdef extern from "PairPrefixSearch.cpp":
    pass

cdef extern from "PairPrefixSearch.h":
    string pair_prefix_search_log(double**, double**, int**, int, int, string)

cdef double** pointer_from_array_double(double [:,:] y):
    cdef int U = y.shape[0]
    cdef double** point_to_y = <double **>malloc(U * sizeof(double*))
    for u in range(U):
        point_to_y[u] = &y[u, 0]
    return point_to_y

cdef int** pointer_from_array_int(int [:,:] y):
    cdef int U = y.shape[0]
    cdef int** point_to_y = <int **>malloc(U * sizeof(int*))
    for u in range(U):
        point_to_y[u] = &y[u, 0]
    return point_to_y

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_forward(y_, label_, alphabet_="ACGT", model_="ctc"):

    cdef int U = y_.shape[0]
    cdef int alphabet_size = y_.shape[1]
    cdef string label = label_.encode("UTF-8")
    cdef string alphabet = alphabet_.encode("UTF-8")
    cdef string model = model_.encode("UTF-8")
    cdef double result

    cdef np.ndarray[double,ndim=2,mode="c"] y = np.asarray(y_, dtype=DTYPE, order="C")
    cdef double** point_to_y = pointer_from_array_double(y)

    try:
        result = forward(&point_to_y[0], U, label, alphabet, model)
        return(result)
    finally:
        free(point_to_y)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_viterbi_acceptor(y_, label_, int band_size=1000, alphabet_="ACGT"):

    cdef int U = y_.shape[0]
    cdef int alphabet_size = y_.shape[1]
    cdef string label = label_.encode("UTF-8")
    cdef string alphabet = alphabet_.encode("UTF-8")

    cdef np.ndarray[double,ndim=2,mode="c"] y = np.asarray(y_, dtype=DTYPE, order="C")
    cdef double** point_to_y = pointer_from_array_double(y)

    try:
        path = viterbi_acceptor_poreover(&point_to_y[0], U, band_size, label, alphabet)
    finally:
        free(point_to_y)

    return(np.array(list(path.decode('utf-8'))).astype(int))

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_beam_search(y_, beam_width_=25, alphabet_="ACGT", model_="ctc"):

    cdef int U = y_.shape[0]
    cdef int alphabet_size = y_.shape[1]
    cdef string alphabet = alphabet_.encode("UTF-8")
    cdef string model = model_.encode("UTF-8")
    cdef int beam_width = beam_width_

    cdef np.ndarray[double,ndim=2,mode="c"] y = np.asarray(y_, dtype=DTYPE, order="C")
    cdef double** point_to_y = pointer_from_array_double(y)

    try:
        decoded_sequence = beam_search(&point_to_y[0], U, alphabet, beam_width, model)
        return(decoded_sequence.decode("UTF-8").lstrip('\x00'))
    finally:
        free(point_to_y)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_beam_search_2d(y1_, y2_, envelope_ranges_=None, beam_width_=25, alphabet_="ACGT", model_="ctc"):

    cdef int U = y1_.shape[0]
    cdef int V = y2_.shape[0]
    cdef int alphabet_size = y1_.shape[1]
    cdef int beam_width = beam_width_
    cdef string alphabet = alphabet_.encode("UTF-8")
    cdef string model = model_.encode("UTF-8")

    cdef np.ndarray[double,ndim=2,mode="c"] y1 = np.asarray(y1_, dtype=DTYPE, order="C")
    cdef np.ndarray[double,ndim=2,mode="c"] y2 = np.asarray(y2_, dtype=DTYPE, order="C")
    cdef np.ndarray[int,ndim=2,mode="c"] envelope_ranges
    if envelope_ranges_ is not None:
        envelope_ranges = np.asarray(envelope_ranges_, dtype=np.intc, order="C")

    cdef double** point_to_y1 = pointer_from_array_double(y1)
    cdef double** point_to_y2 = pointer_from_array_double(y2)
    cdef int** point_to_envelope_ranges
    if envelope_ranges_ is not None:
        point_to_envelope_ranges = pointer_from_array_int(envelope_ranges)

    try:
        if envelope_ranges_ is not None:
            decoded_sequence = beam_search(&point_to_y1[0], &point_to_y2[0], U, V, alphabet, &point_to_envelope_ranges[0], beam_width, model)
        else:
            decoded_sequence = beam_search(&point_to_y1[0], &point_to_y2[0], U, V, alphabet, beam_width, model)
        return(decoded_sequence.decode("UTF-8").lstrip('\x00'))
    finally:
        free(point_to_y1)
        free(point_to_y2)
        if envelope_ranges_ is not None:
            free(point_to_envelope_ranges)

@cython.boundscheck(False)
@cython.wraparound(False)
def cpp_pair_prefix_search_log(y1_, y2_, envelope_ranges_, alphabet_=b'ACGT'):

    cdef int U = y1_.shape[0]
    cdef int V = y2_.shape[0]
    cdef int alphabet_size = y1_.shape[1]
    cdef string alphabet = alphabet_.encode("UTF-8")

    cdef np.ndarray[double,ndim=2,mode="c"] y1 = np.asarray(y1_, dtype=DTYPE, order="C")
    cdef np.ndarray[double,ndim=2,mode="c"] y2 = np.asarray(y2_, dtype=DTYPE, order="C")
    cdef np.ndarray[int,ndim=2,mode="c"] envelope_ranges = np.asarray(envelope_ranges_, dtype=np.intc, order="C")

    cdef double** point_to_y1 = pointer_from_array_double(y1)
    cdef double** point_to_y2 = pointer_from_array_double(y2)
    cdef int** point_to_envelope_ranges = pointer_from_array_int(envelope_ranges)

    try:
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

    cdef np.ndarray[double,ndim=2,mode="c"] y1 = np.asarray(y1_, dtype=DTYPE, order="C")
    cdef np.ndarray[double,ndim=2,mode="c"] y2 = np.asarray(y2_, dtype=DTYPE, order="C")
    cdef np.ndarray[int,ndim=2,mode="c"] envelope_ranges = np.asarray(envelope_ranges_, dtype=np.intc, order="C")

    cdef double** point_to_y1 = pointer_from_array_double(y1)
    cdef double** point_to_y2 = pointer_from_array_double(y2)
    cdef int** point_to_envelope_ranges = pointer_from_array_int(envelope_ranges)

    try:
        testy_mctest = pair_gamma_log_envelope(&point_to_y1[0], &point_to_y2[0], &point_to_envelope_ranges[0], U, V, alphabet_size)
        print(testy_mctest)
    finally:
        free(point_to_y1)
        free(point_to_y2)
        free(point_to_envelope_ranges)
