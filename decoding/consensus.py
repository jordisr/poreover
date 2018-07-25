import numpy as np
import operator, sys
from collections import OrderedDict

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

def forward(l,y):
    '''
    Wrapper for forward_vec that returns full forward matrix.

    Returns matrix of size |L|x|Y| where L is the label and Y are the
    probabilities output by the neural network.

    Not used directly by prefix search.
    '''
    prev = forward_vec(-1, 0, y)
    alpha = np.zeros((len(l)+1,len(y)))
    alpha[0] = prev
    for i,s in enumerate(l):
        prev = forward_vec(s, i+1, y, prev)
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

def prefix_search(y, alphabet=DNA_alphabet):
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

    while not stop_search:
        prefix_prob = {}  # store in dict
        prefix_alphas = []
        search_level += 1

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            alpha_ast = forward_vec_no_gap(prefix_int,y,alpha_prev)
            prefix_prob[prefix] = np.sum(alpha_ast)

            # calculate label probability
            alpha = forward_vec(c_i,search_level,y, previous=alpha_prev)
            label_prob[prefix] = alpha[-1]
            prefix_alphas.append(alpha)

            print(search_level, 'extending by prefix:',c, 'Prefix Probability:',prefix_prob[prefix], 'Label probability:',label_prob[prefix], file=sys.stderr)

        best_prefix = max(prefix_prob.items(), key=operator.itemgetter(1))[0]
        print('best prefix is:',best_prefix, file=sys.stderr)

        if prefix_prob[best_prefix] < label_prob[top_label]:
            stop_search = True
        else:
            # get highest probability label
            top_label = max(label_prob.items(), key=operator.itemgetter(1))[0]
            # then move to prefix with highest prefix probability
            curr_label = best_prefix
            alpha_prev = prefix_alphas[alphabet[curr_label[-1]]]

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
    sys.stderr.write('Calculating gamma...')
    gamma = pair_gamma(y1,y2)
    #gamma, envelope = sample_gamma(y1,y2,50)
    sys.stderr.write('done!\n')

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
        prefix_prob = {}  # store in dict
        prefix_alphas = []
        search_level += 1

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
