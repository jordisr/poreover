'''
Implementations of CTC decoding for PoreOver

This version does not collapse repeated characters so is only valid if the model
has been trained with ctc_merge_repeated=False.
'''

import numpy as np
import operator
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

def forward(l,y):
    t_max = len(y)
    s_max = len(l)

    # fw[t][s] is forward probability of the labeling_{1:s} at time t
    fw = np.zeros((t_max,s_max))

    # fw[-1,-1] = 1 and fw[-1, *] = 0 fw[*,-1] = product of gap probabilites
    for t in range(t_max):
        for s in range(s_max):
            if t==0 and s==0:
                fw[t,s] = y[t,l[s]]
            elif t==0 and s>0:
                fw[t,s] = 0
            elif s==0:
                fw[t,s] = y[t,-1]*fw[t-1,s] + y[t,l[s]]*np.product(y[:t,-1])
            else:
                fw[t,s] = y[t,-1]*fw[t-1,s] + y[t,l[s]]*fw[t-1,s-1]

    return(fw)


# return augmented forward matrix with extra column
def forward_add_column(fw, y, l):
    #print('Adding column for label',l,fw)
    t_max = len(y)
    s_max = len(l)

    fw_ = np.zeros((t_max,s_max))
    fw_[:,:-1] = fw

    s = s_max - 1
    for t in range(t_max):
        if t==0 and s==0:
            fw_[t,s] = y[t,l[s]]
        elif t==0 and s>0:
            fw_[t,s] = 0
        elif s==0:
            fw_[t,s] = y[t,-1]*fw_[t-1,s] + y[t,l[s]]*np.product(y[:t,-1])
        else:
            fw_[t,s] = y[t,-1]*fw_[t-1,s] + y[t,l[s]]*fw_[t-1,s-1]
    return(fw_)

# return prefix probability from the forward matrix
def forward_prefix_prob(fw,y,k):
    if len(k) > 1:
        prefix_prob_ = 0
        for t in range(len(y)):
            if t==0:
                prefix_prob_ += 0
            else:
                prefix_prob_ += y[t,k[-1]]*fw[t-1,-2]
    else:
        prefix_prob_ = np.sum([y[t,k[-1]]*np.product(y[:t,-1]) for t in range(len(y))])
    return(prefix_prob_)

def prefix_search(y, alphabet=OrderedDict([('A',0),('C',1),('G',2),('T',3)]),return_forward=False, return_search=False):
    # y: np array of softmax probabilities
    # alphabet: ordered dict of label to index

    stop_search = False
    search_level = 0

    gap_prob = np.product(y[:,-1])
    label_prob = {'':gap_prob}

    top_label = ''
    top_label_prob = 0
    top_fw = np.array([])

    curr_label = ''
    curr_label_fw = np.array([])

    search_intermediates = []

    while not stop_search:
        #print('Level:',search_level, 'Top Hit:',top_label, 'Label Probability:',label_prob[top_label])
        #print('Current Label:',curr_label)

        prefix_prob = {}
        prefix_fw = []

        for c in alphabet:
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            if search_level == 0:
                prefix_fw.append(forward(prefix_int,y))
            else:
                prefix_fw.append(forward_add_column(curr_label_fw, y, prefix_int))

            label_prob[prefix] = prefix_fw[-1][-1,-1]
            if label_prob[prefix] > top_label_prob:
                top_label = prefix
                top_label_prob = label_prob[prefix]
                top_label_fw = prefix_fw[-1]

            prefix_prob[prefix] = forward_prefix_prob(prefix_fw[-1], y, prefix_int)
            #print('extending by prefix:',c, 'Prefix Probability:',prefix_prob[prefix], 'Label probability:',label_prob[prefix])

        # get best prefix probability
        best_prefix = max(prefix_prob.items(), key=operator.itemgetter(1))[0]

        search_intermediates.append([ (p,prefix_prob[p],label_prob[p]) for p in sorted(prefix_prob.keys())])

        if prefix_prob[best_prefix] < top_label_prob:
            stop_search = True
        else:
            # get highest probability label
            #top_label = max(label_prob.items(), key=operator.itemgetter(1))[0]
            # top_label should be set each iteration, along with top_fw

            # move to prefix with highest label probability
            curr_label = max(prefix_prob.items(), key=operator.itemgetter(1))[0]
            curr_label_fw = prefix_fw[alphabet[curr_label[-1]]]

        search_level += 1

    #print("Search finished! Top label is",top_label, label_prob[top_label])
    if return_forward:
        if return_search:
            return(top_label, top_label_fw, search_intermediates)
        else:
            return(top_label, top_label_fw)
    else:
        return(top_label, label_prob[top_label])
