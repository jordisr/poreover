import numpy as np
import operator, sys
from collections import OrderedDict
from scipy.special import logsumexp

from . import decoding_cy
from heapq import heappush, heappop, heapify

# constants
DNA_alphabet = OrderedDict([('A',0),('C',1),('G',2),('T',3)])
LOG_0 = -float('Inf')
LOG_1 = 0.

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

def prefix_search(y_, alphabet=DNA_alphabet, max_backtrack=0):
    '''
    1. Extend best prefix (remove from heap)
    2. Calculate prefix probabilities of extension (add to heap)
    3. Purge heap of shorter sequences according to max_traceback
    4. Find best prefix
    5. if less than best label, end and return label, otherwise continue
    '''

    y = y_.astype(np.float64)

    # initialize prefix search variables
    stop_search = False
    search_level = 0
    top_label = ''
    curr_label = ''

    curr_label_alphas = []
    gap_prob = np.sum(y[:,-1])
    label_prob = {'':gap_prob}

    longest_prefix = 0

    # initalize variables for 1d forward probabilities
    alpha_prev = decoding_cy.forward_vec_log(-1,search_level,y)

    #top_forward = np.array([])

    prefix_forward = np.zeros(shape=(len(alphabet),len(y),len(y))) + LOG_0

    # priority heap
    prefix_prob = []

    while not stop_search:

        prefix_alphas = []
        search_level += 1

        # prune according to max_backtrack
        prefix_prob = list(filter(lambda x: (len(x[1])) > longest_prefix - max_backtrack, prefix_prob ))

        for c,c_i in alphabet.items():
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            # calculate prefix probability
            alpha_ast = forward_vec_no_gap_log(prefix_int,y,alpha_prev)
            curr_prefix_prob = logsumexp(alpha_ast)

            # calculate label probability
            alpha = decoding_cy.forward_vec_log(c_i,search_level, y, previous=alpha_prev)

            # add to heap
            heappush(prefix_prob,(-curr_prefix_prob, prefix, alpha))

            # label probability from column of forward matrix
            label_prob[prefix] = alpha[-1]

            if label_prob[prefix] > label_prob[top_label]:
                top_label = prefix
                #top_forward = prefix_forward[c_i,:len(prefix)]

            #print(search_level, 'Prefix:', prefix, 'Extending by:',c, 'Prefix_prob:',curr_prefix_prob, 'Label_prob:',label_prob[prefix], file=sys.stderr)

        # get best sequence
        #print('Popping best prefix',prefix_prob[0][1], file=sys.stderr)
        best_prefix_from_heap = heappop(prefix_prob)

        if -best_prefix_from_heap[0] < label_prob[top_label]:
            stop_search = True
        else:
            # move to prefix with highest prefix probability
            curr_label = best_prefix_from_heap[1]
            alpha_prev = best_prefix_from_heap[2]

        if len(curr_label) > longest_prefix:
            longest_prefix = len(curr_label)

    return(top_label, label_prob[top_label])
