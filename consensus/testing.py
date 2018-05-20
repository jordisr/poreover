import itertools
import operator
import numpy as np
from collections import OrderedDict
import consensus

def remove_gaps(a):
    # only needed for greedy decoding
    # unlike standard CTC, does not remove repeated characters
    label = ''
    for i in a:
        if i != '-':
            label += i
    return(label)

class profile:
    '''
    Simple class for probabilistic profiles. Label probabilities are calculated
    by enumerating all paths. Useful for unit testing of CTC code on toy examples.

    Arguments:
    sofmax: numpy array of softmax probabilities, shape is Lx|alphabet|, where L is
        the number of observations
    alphabet: tuple of the character alphabet, including the gap character
    merge_function: function used to map paths to label sequences, this could
        involve collapsing repeated labels or just removing gaps.
    '''
    def __init__(self,softmax, alphabet, merge_function=remove_gaps):
        self.softmax = softmax
        self.alphabet = alphabet
        self.merge_function = merge_function

        self.label_prob = dict()
        self.path_prob = dict()

        for path in itertools.product(range(len(alphabet)),repeat=len(self.softmax)):
            path_prob_ = np.product(self.softmax[np.arange(len(self.softmax)),np.array(path)])
            self.path_prob[path] = path_prob_

            label = self.merge_function([self.alphabet[l] for l in path])
            if label in self.label_prob:
                self.label_prob[label] += path_prob_
            else:
                self.label_prob[label] = path_prob_

    def top_label_prob(self):
        for k,v in reversed(sorted(self.label_prob.items(), key=operator.itemgetter(1), reverse=False)[-20:]):
            print(k,v)

    def prefix_prob(self,prefix):
        prefix_prob_ = 0
        for t in range(len(self.softmax)):
            for path in itertools.product(range(len(self.alphabet)),repeat=(t+1)):
                path_prob_ = np.product(self.softmax[np.arange(t+1),np.array(path)])
                label = self.merge_function([self.alphabet[l] for l in path])
                if label == prefix:
                    if path[-1] != 2:
                        prefix_prob_ += path_prob_
        return(prefix_prob_)

def joint_prob(profile1, profile2):
    joint_label_prob = dict()
    for k,v1 in profile1.label_prob.items():
        if k in profile2.label_prob:
            v2 = profile2.label_prob[k]
            joint_label_prob[k] = v1*v2
        else:
            joint_label_prob[k] = 0
    return(joint_label_prob)

def test_pair_forward(y1,y2, examples,envelope=None,forward_algorithm=consensus.pair_forward):
    alphabet = ('A','B','')
    alphabet_dict = {'A':0,'B':1,'':2}

    profile1=profile(y1,alphabet,remove_gaps)
    profile2=profile(y2,alphabet,remove_gaps)
    joint_label_prob = joint_prob(profile1,profile2)

    for label in examples:
        label_int = [alphabet_dict[i] for i in label]
        alpha,_,_  = forward_algorithm(label_int,y1,y2,mask=envelope)
        print(label,consensus.pair_label_prob(alpha), joint_label_prob[label])

def test_prefix_search(y1,y2,envelope=None):
    alphabet = ('A','B','')
    toy_alphabet = OrderedDict([('A',0),('B',1)])

    profile1=profile(y1,alphabet,remove_gaps)
    profile2=profile(y2,alphabet,remove_gaps)
    joint_label_prob = joint_prob(profile1,profile2)

    top_label = max(joint_label_prob.items(), key=operator.itemgetter(1))[0]
    print('top_label:',top_label, 'probability:',joint_label_prob[top_label],
    'prefix_search:',consensus.pair_prefix_search(y1,y2,alphabet=toy_alphabet, envelope=envelope))

if __name__ == '__main__':

    print('--- Testing pair prefix search ---')
    y1 = y2 = np.array([[0.1,0.6,0.3],[0.4,0.2,0.4],[0.4,0.3,0.3],[0.2,0.8,0]])
    test_prefix_search(y1,y2)
    y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
    y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
    test_prefix_search(y1,y2)
    y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
    y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5]])
    test_prefix_search(y1,y2)

    print('--- Testing forward algorithm ---')
    y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
    y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
    examples = ['AAAA','ABBA','ABA','AA','BB','A','B']
    test_pair_forward(y1,y2,examples=examples)

    print('--- Testing forward algorithm with full alignment envelope ---')
    U = len(y1)
    V = len(y2)
    full_envelope = consensus.alignment_envelope_dense(U,V)
    for u in range(U):
        for v in range(V):
            full_envelope.add(u,v)
    test_pair_forward(y1,y2,examples=examples,envelope=full_envelope)

    print('alternative implementation (should be the same)')
    test_pair_forward(y1,y2,examples=examples,envelope=full_envelope,forward_algorithm=consensus.pair_forward_sparse)

    print('--- Testing banded alignment envelope ---')
    (U, V) = (5,5)
    print('width=0\n',consensus.diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',consensus.diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',consensus.diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',consensus.diagonal_band_envelope(U,V,3).toarray())

    (U, V) = (5,4)
    print('width=0\n',consensus.diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',consensus.diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',consensus.diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',consensus.diagonal_band_envelope(U,V,3).toarray())

    (U, V) = (2,10)
    print('width=0\n',consensus.diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',consensus.diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',consensus.diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',consensus.diagonal_band_envelope(U,V,3).toarray())

    print('--- Single diagonal band doesn\'t match ---')
    y1 = y2 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.8,0.1,0.1]])
    (U, V) = (len(y1),len(y2))
    band_envelope = consensus.diagonal_band_envelope(U,V,0)
    test_pair_forward(y1,y2,examples=examples,envelope=band_envelope)
    test_prefix_search(y1,y2,envelope=band_envelope)

    print('except when most of the probability passes through it')
    y1 = y2 = np.array([[1,0,0],[0,1,0]])
    (U, V) = (len(y1),len(y2))
    band_envelope = consensus.diagonal_band_envelope(U,V,0)
    test_prefix_search(y1,y2,envelope=band_envelope)
