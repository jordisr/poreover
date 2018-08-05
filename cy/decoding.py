import numpy as np
import operator, sys
from collections import OrderedDict
from scipy.special import logsumexp

import pyximport; pyximport.install()
from . import decoding_cy

# Default alphabet
DNA_alphabet = OrderedDict([('A',0),('C',1),('G',2),('T',3)])

def remove_gaps(a):
    # only needed for greedy decoding
    # unlike standard CTC, does not remove repeated characters
    label = ''
    for i in a:
        if i != '-':
            label += i
    return(label)

def greedy_search(logits, alphabet=['A','C','G','T','-']):
    # take highest probability label at each step
    argmax_ = np.argmax(logits, axis=1)
    chars_ = np.take(alphabet, argmax_)
    return(remove_gaps(chars_))

# identities for log scale calculations
LOG_0 = -float('Inf')
LOG_1 = 0.

def pair_gamma(y1,y2):
    '''
    Finds gamma matrix, where gamma[0,0] is the probability that both RNNs
    agree on the same sequence. Returns a dense matrix of shape UxV.
    '''
    U = len(y1)
    V = len(y2)

    # intialization
    gamma_ = np.zeros(shape=(U+1,V+1))
    gamma_ast = np.zeros(shape=(U+1,V+1))
    gamma_[U,V] = 1
    gamma_ast[U,V] = 1

    for v in range(V):
        gamma_[U,v] = np.prod(y2[v:,-1])
    for u in range(U):
        gamma_[u,V] = np.prod(y1[u:,-1])

    for u in reversed(range(U)):
        for v in reversed(range(V)):
            # individual recursions
            gamma_eps = gamma_[u+1,v]*y1[u,-1]
            gamma_ast_eps = gamma_ast[u,v+1]*y2[v,-1]
            gamma_ast_ast = gamma_[u+1,v+1]*np.dot(y1[u,:-1],y2[v,:-1])

            # storing DP matrices
            gamma_ast[u,v] = gamma_ast_eps + gamma_ast_ast
            gamma_[u,v] = gamma_eps + gamma_ast[u,v]

    return(gamma_)

def pair_gamma_log(y1,y2):
    '''
    Finds gamma matrix, where gamma[0,0] is the probability that both RNNs
    agree on the same sequence. Returns a dense matrix of shape UxV.
    '''
    U = len(y1)
    V = len(y2)

    # intialization
    gamma_ = np.zeros(shape=(U+1,V+1)) + LOG_0
    gamma_ast = np.zeros(shape=(U+1,V+1)) + LOG_0
    gamma_[U,V] = LOG_1
    gamma_ast[U,V] = LOG_1

    for v in range(V):
        gamma_[U,v] = np.sum(y2[v:,-1])
    for u in range(U):
        gamma_[u,V] = np.sum(y1[u:,-1])

    for u in reversed(range(U)):
        for v in reversed(range(V)):
            # individual recursions
            gamma_eps = gamma_[u+1,v] + y1[u,-1]
            gamma_ast_eps = gamma_ast[u,v+1] + y2[v,-1]
            gamma_ast_ast = gamma_[u+1,v+1] + logsumexp(y1[u,:-1]+y2[v,:-1])

            # storing DP matrices
            gamma_ast[u,v] = np.logaddexp(gamma_ast_eps, gamma_ast_ast)
            gamma_[u,v] = np.logaddexp(gamma_eps, gamma_ast[u,v])

    return(gamma_)

def forward_vec(s,i,y,previous=None):
    '''
    Arguments:
        s: character index
        i: label index
        y: softmax probabilities
        previous: last column
    1d forward algorithm
    Just calculates one column on the character s.
    '''
    t_max = len(y)
    fw = np.zeros(t_max)
    assert(i==0 or previous is not None)
    for t in range(t_max):
        if i==0:
            if t==0:
                fw[t] = y[t,s]
            else:
                fw[t] = y[t,-1]*fw[t-1]
        elif t==0:
            if i==1:
                fw[t] = y[t,s]
        else:
            fw[t] = y[t,-1]*fw[t-1] + y[t,s]*previous[t-1]
    return(fw)

def forward(l,y,fw_fn=forward_vec):
    '''
    Wrapper for forward_vec that returns full forward matrix.

    Returns matrix of size |L|x|Y| where L is the label and Y are the
    probabilities output by the neural network.

    Not used directly by prefix search.
    '''
    prev = fw_fn(-1, 0, y)
    alpha = np.zeros((len(l)+1,len(y)))
    alpha[0] = prev
    for i,s in enumerate(l):
        prev = fw_fn(s, i+1, y, prev)
        alpha[i+1] = prev
    return(alpha)

def forward_vec_no_gap(l,y,fw0):
    '''
    Forward variable of paths that do not end on a gap
    '''
    try:
        if (len(l) == 1):
            return(np.insert(fw0[:-1],0,1)*y[:,l[-1]])
        else:
            #return(fw0[:-1]*y[1:,l[-1]])
            return(np.insert(fw0[:-1],0,0)*y[:,l[-1]])
    except IndexError:
        print('Ran into IndexError!', file=sys.stderr)
        return 1

def prefix_search(y, alphabet=DNA_alphabet, return_forward=False):
    '''
    1D prefix search
    '''

    # initialize prefix search variables
    stop_search = False
    search_level = 0
    top_label = ''
    curr_label = ''
    curr_label_alphas = []
    gap_prob = np.product(y[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    alpha_prev = forward_vec(-1,search_level,y)
    top_forward = np.array([])
    prefix_forward = np.zeros(shape=(len(alphabet),len(y),len(y))) # TESTING

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []
        search_level += 1

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]
            if c_i == 0:
                best_prefix = prefix

            alpha_ast = forward_vec_no_gap(prefix_int,y,alpha_prev)
            prefix_prob[prefix] = np.sum(alpha_ast)

            # calculate label probability
            alpha = forward_vec(c_i,search_level,y, previous=alpha_prev)
            prefix_forward[c_i, search_level-1] = alpha
            label_prob[prefix] = alpha[-1]
            if label_prob[prefix] > label_prob[top_label]:
                top_label = prefix
                top_forward = prefix_forward[c_i,:len(prefix)]
                #print(len(top_label),len(top_forward))
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

def forward_vec_no_gap_log(l,y,fw0):
    '''
    Forward variable of paths that do not end on a gap
    '''
    try:
        if (len(l) == 1):
            return(np.insert(fw0[:-1],0,LOG_1) + y[:,l[-1]])
        else:
            #return(fw0[:-1]*y[1:,l[-1]])
            return(np.insert(fw0[:-1],0,LOG_0) + y[:,l[-1]])
    except IndexError:
        print('Ran into IndexError!', file=sys.stderr)
        return 1

'''
def forward_vec_log(s,i,y,previous=None):
    t_max = len(y)
    fw = np.zeros(t_max) + LOG_0
    assert(i==0 or previous is not None)
    for t in range(t_max):
        if i==0:
            if t==0:
                fw[t] = y[t,s]
            else:
                fw[t] = y[t,-1] + fw[t-1]
        elif t==0:
            if i==1:
                fw[t] = y[t,s]
        else:
            fw[t] = np.logaddexp(y[t,-1]+fw[t-1], y[t,s]+previous[t-1])
    return(fw)
'''

def prefix_search_log(y_, alphabet=DNA_alphabet, return_forward=False):

    y = y_.astype(np.float64)

    # initialize prefix search variables
    stop_search = False
    search_level = 0
    top_label = ''
    curr_label = ''
    curr_label_alphas = []
    gap_prob = np.sum(y[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    alpha_prev = decoding_cy.forward_vec_log(-1,search_level,y)
    #print(alpha_prev)
    top_forward = np.array([])
    prefix_forward = np.zeros(shape=(len(alphabet),len(y),len(y))) + LOG_0

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
            alpha = decoding_cy.forward_vec_log(c_i,search_level,y, previous=alpha_prev)
            prefix_forward[c_i, search_level-1] = alpha
            label_prob[prefix] = alpha[-1]
            if label_prob[prefix] > label_prob[top_label]:
                top_label = prefix
                top_forward = prefix_forward[c_i,:len(prefix)]
                #print(len(top_label),len(top_forward))
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

def pair_prefix_prob(alpha_ast_ast,gamma, envelope=None):
    U,V = alpha_ast_ast.shape
    prefix_prob = 0
    if envelope == None:
        prefix_prob = np.sum(alpha_ast_ast*gamma[1:,1:])
    else:
        for k in envelope.keys():
            (u,v) = k
            if u < U and v < V:
                prefix_prob += alpha_ast_ast[u,v]*gamma[u,v]
    return(prefix_prob / gamma[0,0])

def pair_prefix_search(y1, y2, alphabet=DNA_alphabet):
    '''
    Do 2d prefix search. Arguments are softmax probabilities of each read,
    an alignment_envelope object, and an OrderedDict with the alphabet.
    Tries to be more clever about vectorization and not iterating over
    full alpha 2d matrix.
    '''

    # calculate full gamma matrix
    gamma = decoding_pair_gamma(y1,y2)

    # initialize prefix search variables
    stop_search = False
    search_level = 0
    top_label = ''
    curr_label = ''
    curr_label_alphas = []
    gap_prob = np.product(y1[:,-1])*np.product(y2[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    alpha1_prev = forward_vec(-1,search_level,y1)
    alpha2_prev = forward_vec(-1,search_level,y2)
    alpha_ast1 = np.array([])
    alpha_ast2 = np.array([])

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []
        search_level += 1

        if len(curr_label) > max(len(y1),len(y2)):
            stop_search = True
            print('WARNING: Max search depth exceeded while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            # calculate prefix probability with outer product
            alpha_ast1 = forward_vec_no_gap(prefix_int,y1,alpha1_prev)
            alpha_ast2 = forward_vec_no_gap(prefix_int,y2,alpha2_prev)
            alpha_ast_ast = np.outer(alpha_ast1,alpha_ast2)
            prefix_prob[prefix] = pair_prefix_prob(alpha_ast_ast, gamma)

            # calculate label probability
            alpha1 = forward_vec(c_i,search_level,y1, previous=alpha1_prev)
            alpha2 = forward_vec(c_i,search_level,y2, previous=alpha2_prev)
            label_prob[prefix] = alpha1[-1]*alpha2[-1]/gamma[0,0]
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

def pair_prefix_prob_log(alpha_ast_ast, gamma, envelope=None):
    U,V = alpha_ast_ast.shape
    prefix_prob = LOG_0
    if envelope == None:
        prefix_prob = logsumexp((alpha_ast_ast+gamma[1:,1:]).flatten())
    return(prefix_prob - gamma[0,0])

def pair_prefix_search_log(y1_, y2_, alphabet=DNA_alphabet):
    '''
    Do 2d prefix search. Arguments are softmax probabilities of each read,
    an alignment_envelope object, and an OrderedDict with the alphabet.
    Tries to be more clever about vectorization and not iterating over
    full alpha 2d matrix.
    '''
    y1 = y1_.astype(np.float64)
    y2 = y2_.astype(np.float64)

    gamma = decoding_cy.pair_gamma_log(y1,y2)

    # initialize prefix search variables
    stop_search = False
    search_level = 0
    top_label = ''
    curr_label = ''
    curr_label_alphas = []
    gap_prob = np.sum(y1[:,-1]) + np.sum(y2[:,-1])
    label_prob = {'':gap_prob}

    # initalize variables for 1d forward probabilities
    alpha1_prev = decoding_cy.forward_vec_log(-1,search_level,y1)
    alpha2_prev = decoding_cy.forward_vec_log(-1,search_level,y2)
    alpha_ast1 = np.array([])
    alpha_ast2 = np.array([])

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []
        search_level += 1

        if len(curr_label) > max(len(y1),len(y2)):
            stop_search = True
            print('WARNING: Max search depth exceeded while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            # calculate prefix probability with outer product
            alpha_ast1 = forward_vec_no_gap_log(prefix_int,y1,alpha1_prev)
            alpha_ast2 = forward_vec_no_gap_log(prefix_int,y2,alpha2_prev)
            alpha_ast_ast = np.add.outer(alpha_ast1,alpha_ast2)
            prefix_prob[prefix] = pair_prefix_prob_log(alpha_ast_ast, gamma)

            # calculate label probability
            alpha1 = decoding_cy.forward_vec_log(c_i,search_level,y1, previous=alpha1_prev)
            alpha2 = decoding_cy.forward_vec_log(c_i,search_level,y2, previous=alpha2_prev)
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
