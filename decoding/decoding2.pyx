# cython: infer_types=True
cimport cython
cimport numpy as np
from numpy cimport ndarray
import numpy as np
import operator, sys

cdef extern from "math.h":
    double log(double m)
    double exp(double m)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double LOG_0 = -9999
cdef double LOG_1 = 0

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

@cython.boundscheck(False)
@cython.wraparound(False)
def pair_prefix_prob_log_from_vec(ndarray[DTYPE_t, ndim=1] alpha_ast1, ndarray[DTYPE_t, ndim=1] alpha_ast2, ndarray[DTYPE_t, ndim=2] gamma):
    cdef Py_ssize_t U = alpha_ast1.shape[0]
    cdef Py_ssize_t V = alpha_ast2.shape[0]
    cdef Py_ssize_t u, v
    cdef double prefix_prob = 0
    for u in range(U):
        for v in range(V):
            prefix_prob += exp(alpha_ast1[u] + alpha_ast2[v] + gamma[u+1,v+1])
    return(log(prefix_prob) - gamma[0,0])

# doesn't seem to be faster than vectorized numpy
@cython.boundscheck(False)
@cython.wraparound(False)
def pair_prefix_prob_log(double [:,:] alpha_ast_ast, double [:,:] gamma):
    cdef Py_ssize_t U = alpha_ast_ast.shape[0]
    cdef Py_ssize_t V = alpha_ast_ast.shape[1]
    cdef Py_ssize_t u, v
    cdef double prefix_prob = 0
    for v in range(V):
        for u in range(U):
            prefix_prob += exp(alpha_ast_ast[u,v] + gamma[u+1,v+1])
    return(log(prefix_prob) - gamma[0,0])

###################### LESS CYTHONIC CODE HERE ##################################

# Default alphabet
from collections import OrderedDict
DNA_alphabet = OrderedDict([('A',0),('C',1),('G',2),('T',3)])

def forward_vec_no_gap_log(l,y,fw0):
    '''
    Forward variable of paths that do not end on a gap
    '''
    if (len(l) == 1):
        return(np.insert(fw0[:-1],0,LOG_1) + y[:,l[-1]])
    else:
        return(np.insert(fw0[:-1],0,LOG_0) + y[:,l[-1]])

def prefix_search_log(y_, alphabet=DNA_alphabet, return_forward=False):

    cdef np.ndarray[DTYPE_t, ndim=2] y = y_.astype(np.float64)

    # initialize prefix search variables
    stop_search = False
    cdef int search_level = 0
    cdef str top_label = ''
    cdef str curr_label = ''
    cdef float gap_prob = sum(y[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    cdef np.ndarray[DTYPE_t, ndim=1] alpha_prev = forward_vec_log(-1,search_level,y)
    cdef np.ndarray[DTYPE_t, ndim=2] top_forward
    cdef np.ndarray[DTYPE_t, ndim=3] prefix_forward = np.zeros(shape=(len(alphabet),len(y),len(y))) + LOG_0

    # iterators
    #cdef char c
    cdef int c_i
    #cdef str prefix, best_prefix
    cdef np.ndarray[DTYPE_t, ndim=1] alpha, alpha_ast

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []
        search_level += 1

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]
            if c_i == 0:
                best_prefix = prefix

            alpha_ast = forward_vec_no_gap_log(prefix_int,y,alpha_prev)
            prefix_prob[prefix] = logsumexp(alpha_ast)

            # calculate label probability
            alpha = forward_vec_log(c_i,search_level,y, previous=alpha_prev)
            prefix_forward[c_i, search_level-1] = alpha
            label_prob[prefix] = alpha[-1]
            if label_prob[prefix] > label_prob[top_label]:
                top_label = prefix
                top_forward = prefix_forward[c_i,:len(prefix)]
            if prefix_prob[prefix] > prefix_prob[best_prefix]:
                best_prefix = prefix
            prefix_alphas.append(alpha)

            #print(search_level, 'extending by prefix:',c, 'Prefix Probability:',prefix_prob[prefix], 'Label probability:',label_prob[prefix], file=sys.stderr)

        #best_prefix = max(prefix_prob.items(), key=operator.itemgetter(1))[0]
        #print('best prefix is:',best_prefix, file=sys.stderr)

        if prefix_prob[best_prefix] < label_prob[top_label]:
            stop_search = True
        else:
            # get highest probability label
            #top_label = max(label_prob.items(), key=operator.itemgetter(1))[0]
            # then move to prefix with highest prefix probability
            curr_label = best_prefix
            alpha_prev = prefix_alphas[alphabet[curr_label[-1]]]

    if return_forward:
        return(top_label, top_forward.T)
    else:
        return(top_label, label_prob[top_label])

def pair_prefix_search_log(y1_, y2_, alphabet=DNA_alphabet):
    '''
    Do 2d prefix search. Arguments are softmax probabilities of each read,
    an alignment_envelope object, and an OrderedDict with the alphabet.
    Tries to be more clever about vectorization and not iterating over
    full alpha 2d matrix.
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] y1 = y1_.astype(np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] y2 = y2_.astype(np.float64)

    cdef Py_ssize_t U = y1_.shape[0]
    cdef Py_ssize_t V = y2_.shape[0]

    gamma = pair_gamma_log(y1,y2)

    # initialize prefix search variables
    stop_search = False
    cdef int search_level = 0
    cdef str top_label = ''
    cdef str curr_label = ''
    cdef double gap_prob = np.sum(y1[:,-1]) + np.sum(y2[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    cdef np.ndarray[DTYPE_t, ndim=1] alpha1_prev = forward_vec_log(-1,search_level,y1)
    cdef np.ndarray[DTYPE_t, ndim=1] alpha2_prev = forward_vec_log(-1,search_level,y2)
    cdef np.ndarray[DTYPE_t, ndim=1] alpha_ast1, alpha_ast2
    cdef np.ndarray[DTYPE_t, ndim=1] alpha1, alpha2

    cdef str prefix, best_prefix

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []
        search_level += 1

        if len(curr_label) > max(U,V):
            stop_search = True
            print('WARNING: Max search depth exceeded!')

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            # calculate prefix probability with outer product
            alpha_ast1 = forward_vec_no_gap_log(prefix_int,y1,alpha1_prev)
            alpha_ast2 = forward_vec_no_gap_log(prefix_int,y2,alpha2_prev)

            #prefix_prob[prefix] = pair_prefix_prob_log_from_vec(alpha_ast1,alpha_ast2,gamma)
            unrolled = (np.add.outer(alpha_ast1,alpha_ast2)+gamma[1:,1:]).flatten()
            #prefix_prob[prefix] = np.max(unrolled) + np.log(len(unrolled)) - gamma[0,0]
            prefix_prob[prefix] = logsumexp(unrolled) - gamma[0,0]

            # calculate label probability
            alpha1 = forward_vec_log(c_i,search_level,y1, previous=alpha1_prev)
            alpha2 = forward_vec_log(c_i,search_level,y2, previous=alpha2_prev)
            label_prob[prefix] = alpha1[-1] + alpha2[-1] - gamma[0,0]
            prefix_alphas.append((alpha1,alpha2))
            #print(search_level, 'extending by prefix:',c, 'Prefix Probability:',prefix_prob[prefix], 'Label probability:',label_prob[prefix], file=sys.stderr)

        best_prefix = max(prefix_prob.items(), key=operator.itemgetter(1))[0]
        #print('best prefix is:',best_prefix, file=sys.stderr)

        if prefix_prob[best_prefix] < label_prob[top_label]:
            stop_search = True
        else:
            # get highest probability label
            top_label = max(label_prob.items(), key=operator.itemgetter(1))[0]
            # then move to prefix with highest prefix probability
            curr_label = best_prefix
            (alpha1_prev, alpha2_prev) = prefix_alphas[alphabet[curr_label[-1]]]

    return(top_label, label_prob[top_label])

###################### NON-CYTHONIC CODE HERE ##################################
