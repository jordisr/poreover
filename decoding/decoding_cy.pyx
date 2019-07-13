# cython: infer_types=True, language_level=2
cimport cython
cimport numpy as np
import numpy as np
import copy
from libcpp.string cimport string
#from scipy.special import logsumexp

from SparseMatrix cimport SparseMatrix

cdef extern from "math.h":
    double log(double m)
    double exp(double m)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double LOG_0 = -9999
cdef double LOG_1 = 0

cdef class PySparseMatrix:
    cdef SparseMatrix*mat  # Hold a C++ instance which we're wrapping

    def __cinit__(self):
        self.mat =  new SparseMatrix()

    def __dealloc__(self):
        del self.mat

    def get(self, int i, int j):
        return self.mat.get(i,j)

    def set(self, int i, int j, double value):
        self.mat.set(i, j, value)
        return True

    def push_row(self, int start, int end):
        self.mat.push_row(start, end)
        return True

def diagonal_band_envelope(U,V,width,inside=1,outside=0):
    # just steps across main diagonal line. Width is size above and below
    # the main diagonal
    envelope = PySparseMatrix()
    envelope_indices = []
    envelope_ranges = []
    for u in range(U):
        center = int(np.round(V/U*u))
        start = max(center-width,0)
        end = min(center+width,V-1)
        envelope.push_row(start,end)
        envelope_ranges.append((start,end))
        for v in range(start,end+1):
            envelope.set(u,v,inside)
            envelope_indices.append((u,v))
    return(envelope, np.array(envelope_ranges), np.array(envelope_indices))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)   # Deactivate negative indexing.
def viterbi_acceptor(double [:,:] y, label_, alphabet='ACGT', band_size=0):
    cdef Py_ssize_t t_max = y.shape[0]
    cdef Py_ssize_t l_max = len(label_)
    cdef int gap_char = len(alphabet)

    # convert label to alphabet indices
    alphabet_dict = {}
    for i, c in enumerate(alphabet):
        alphabet_dict[c] = i
    cdef long [:] label = np.array([alphabet_dict[c] for c in label_]).astype(np.int64)

    v_np = np.zeros((l_max+1, t_max))-np.inf
    cdef double [:, :] v = v_np

    ptr_np = np.zeros_like(v).astype(np.int64) + gap_char
    cdef long [:, :] ptr = ptr_np

    cdef Py_ssize_t t
    cdef Py_ssize_t l

    cdef double emit_prob
    cdef double stay_prob
    cdef double gap_prob

    # optionally band Viterbi DP
    cdef int band_size_
    if band_size > 0:
        band_size_ = band_size
    else:
        band_size_ = t_max

    # initialize DP matrix
    for t in range(t_max):
        gap_prob = 0
        for t_ in range(t+1):
            gap_prob += y[t_, gap_char]
        v[0, t] = gap_prob
        ptr[0, t] = gap_char

    v[1,0] = y[0,label[0]]
    ptr[0,0] = 1

    # fill out DP matrix
    for l in range(1,l_max+1):
        for t in range(max(1, int(l/l_max*t)-band_size_),min(t_max, int(l/l_max*t)+band_size_)):
            if t >= l:
                emit_prob = y[t, label[l-1]] + v[l-1,t-1]
                stay_prob = y[t, gap_char] + v[l,t-1]
                if emit_prob > stay_prob:
                    v[l,t] = emit_prob
                    ptr[l,t] = 1
                else:
                    v[l,t] = stay_prob
                    ptr[l,t] = 0

    # traceback
    path = np.zeros(t_max).astype(int) + gap_char
    l = len(label)
    t = t_max-1
    while (l > 0):
        if ptr[l,t]:
            path[t] = label[l-1]
            l -= 1
        t -= 1
    return path

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)   # Deactivate negative indexing.
def forward_vec_log(int s, int i, double [:,:] y, double [:] previous=None):
    '''
    Arguments:
        s: character index
        i: label index
        y: softmax probabilities
        previous: last column
    1d forward algorithm: Just calculates one column on the character s.
    '''

    cdef Py_ssize_t t_max = y.shape[0]
    cdef Py_ssize_t t

    fw_np = np.zeros(t_max, dtype=DTYPE) + LOG_0
    cdef double [:] fw = fw_np

    assert(i==0 or previous is not None)

    for t in range(t_max):
        if i==0:
            if t==0:
                fw[t] = y[t,s]
            else:
                fw[t] = y[t,-1]+fw[t-1]
        elif t==0:
            if i==1:
                fw[t] = y[t,s]
        else:
            fw[t] = log(exp(y[t,-1]+fw[t-1]) + exp(y[t,s]+previous[t-1]))
    return(fw_np)

# define optmized math functions using exp and log from math.h
cdef double sum(double [:] x):
    cdef Py_ssize_t x_max = x.shape[0]
    cdef double total = 0
    cdef Py_ssize_t i
    for i in range(x_max):
        total += x[i]
    return(total)

cpdef double logsumexp(double [:] x):
    cdef Py_ssize_t x_max = x.shape[0]
    cdef double total = 0
    cdef Py_ssize_t i
    for i in range(x_max):
        total += exp(x[i])
    return(log(total))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pair_gamma_log(double [:,:] y1, double [:,:] y2):

    cdef Py_ssize_t U = y1.shape[0]
    cdef Py_ssize_t V = y2.shape[0]
    cdef Py_ssize_t alphabet_size = y1.shape[1]
    cdef Py_ssize_t u, v, t

    # intialization
    cdef np.ndarray[DTYPE_t, ndim=2] gamma_np = np.zeros(shape=(U+1,V+1), dtype=DTYPE) + LOG_0
    cdef double [:,:] gamma_ = gamma_np

    cdef np.ndarray[DTYPE_t, ndim=2] gamma_ast_np = np.zeros(shape=(U+1,V+1), dtype=DTYPE) + LOG_0
    cdef double [:,:] gamma_ast = gamma_ast_np
    cdef double gamma_eps, gamma_ast_eps, gamma_ast_ast

    gamma_[U,V] = LOG_1
    gamma_ast[U,V] = LOG_1

    for v in range(V):
        gamma_[U,v] = sum(y2[v:,alphabet_size-1])
    for u in range(U):
        gamma_[u,V] = sum(y1[u:,alphabet_size-1])

    for u in reversed(range(U)):
        for v in reversed(range(V)):
            # individual recursions
            gamma_eps = gamma_[u+1,v] + y1[u,alphabet_size-1]
            gamma_ast_eps = gamma_ast[u,v+1] + y2[v,alphabet_size-1]

            # method 1
            #total1 = logsumexp(np.add(y1[u,:alphabet_size-1], y2[v,:alphabet_size-1]))

            # method 2
            total2 = 0
            for t in range(0,alphabet_size-1):
                total2 += exp(y1[u,t]+y2[v,t])

            gamma_ast_ast = gamma_[u+1,v+1] + log(total2)

            # storing DP matrices
            gamma_ast[u,v] = log(exp(gamma_ast_eps) + exp(gamma_ast_ast))
            gamma_[u,v] = log(exp(gamma_eps) + exp(gamma_ast[u,v]))

    return(gamma_np)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pair_gamma_log_envelope(double [:,:] y1, double [:,:] y2, PySparseMatrix envelope, Py_ssize_t [:,:] envelope_indices, PySparseMatrix gamma_, PySparseMatrix gamma_ast):

    cdef Py_ssize_t U = y1.shape[0]
    cdef Py_ssize_t V = y2.shape[0]
    cdef Py_ssize_t alphabet_size = y1.shape[1]
    cdef Py_ssize_t u, v, t
    cdef Py_ssize_t[:] tup

    # intialization
    #gamma_ = copy.deepcopy(envelope)
    #gamma_ast = copy.deepcopy(envelope)

    #gamma_ = envelope
    #gamma_ast = envelope

    cdef double gamma_eps, gamma_ast_eps, gamma_ast_ast

    gamma_.set(U,V,LOG_1)
    gamma_ast.set(U,V,LOG_1)

    for v in range(V):
        gamma_.set(U,v,sum(y2[v:,alphabet_size-1]))
    for u in range(U):
        gamma_.set(u,V,sum(y1[u:,alphabet_size-1]))

    for tup in reversed(envelope_indices):
        u = tup[0]
        v = tup[1]
        if u < U and v < V:
            # individual recursions
            gamma_eps = gamma_.get(u+1,v) + y1[u,alphabet_size-1]
            gamma_ast_eps = gamma_ast.get(u,v+1) + y2[v,alphabet_size-1]

            # method 1
            #total1 = logsumexp(np.add(y1[u,:alphabet_size-1], y2[v,:alphabet_size-1]))

            # method 2
            total2 = 0
            for t in range(0,alphabet_size-1):
                total2 += exp(y1[u,t]+y2[v,t])

            gamma_ast_ast = gamma_.get(u+1,v+1) + log(total2)

            # storing DP matrices
            gamma_ast.set(u,v,log(exp(gamma_ast_eps) + exp(gamma_ast_ast)))
            gamma_.set(u,v,log(exp(gamma_eps) + exp(gamma_ast.get(u,v))))

    return(gamma_)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pair_gamma_log_envelope2(double [:,:] y1, double [:,:] y2, envelope):
    '''
    Older version that uses Python sparse envelope class, much slower
    '''

    cdef Py_ssize_t U = y1.shape[0]
    cdef Py_ssize_t V = y2.shape[0]
    cdef Py_ssize_t alphabet_size = y1.shape[1]
    cdef Py_ssize_t u, v, t

    # intialization
    gamma_ = copy.deepcopy(envelope)
    gamma_ast = copy.deepcopy(envelope)

    cdef double gamma_eps, gamma_ast_eps, gamma_ast_ast

    gamma_[U,V] = LOG_1
    gamma_ast[U,V] = LOG_1

    for v in range(V):
        if (U,v) in envelope:
            gamma_[U,v] = sum(y2[v:,alphabet_size-1])
    for u in range(U):
        if (u,V) in envelope:
            gamma_[u,V] = sum(y1[u:,alphabet_size-1])

    for (u,v) in reversed(envelope.indices):
        if u < U and v < V:
            # individual recursions
            gamma_eps = gamma_[u+1,v] + y1[u,alphabet_size-1]
            gamma_ast_eps = gamma_ast[u,v+1] + y2[v,alphabet_size-1]

            # method 1
            #total1 = logsumexp(np.add(y1[u,:alphabet_size-1], y2[v,:alphabet_size-1]))

            # method 2
            total2 = 0
            for t in range(0,alphabet_size-1):
                total2 += exp(y1[u,t]+y2[v,t])

            gamma_ast_ast = gamma_[u+1,v+1] + log(total2)

            # storing DP matrices
            gamma_ast[u,v] = log(exp(gamma_ast_eps) + exp(gamma_ast_ast))
            gamma_[u,v] = log(exp(gamma_eps) + exp(gamma_ast[u,v]))

    return(gamma_)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pair_prefix_prob_log_from_vec(double [:] alpha_ast1, double [:] alpha_ast2, double [:,:] gamma):
    cdef Py_ssize_t U = alpha_ast1.shape[0]
    cdef Py_ssize_t V = alpha_ast2.shape[0]
    cdef Py_ssize_t u, v
    cdef double prefix_prob = 0
    for u in range(U):
        for v in range(V):
            prefix_prob += exp(alpha_ast1[u] + alpha_ast2[v] + gamma[u+1,v+1])
    return(log(prefix_prob) - gamma[0,0])

# doesn't seem to be faster than vectorized numpy
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pair_prefix_prob_log(double [:,:] alpha_ast_ast, double [:,:] gamma):
    cdef Py_ssize_t U = alpha_ast_ast.shape[0]
    cdef Py_ssize_t V = alpha_ast_ast.shape[1]
    cdef Py_ssize_t u, v
    cdef double prefix_prob = 0
    for v in range(V):
        for u in range(U):
            prefix_prob += exp(alpha_ast_ast[u,v] + gamma[u+1,v+1])
    return(log(prefix_prob) - gamma[0,0])
