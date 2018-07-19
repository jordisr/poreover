import itertools
import operator
import numpy as np
from collections import OrderedDict

import consensus
from ctc import remove_gaps

class profile:
    '''
    Simple class for probabilistic profiles. Label probabilities are calculated
    by enumerating all paths. Useful for unit testing of CTC code on toy examples.

    Arguments:
    sofmax: numpy array of softmax probabilities, shape is Lx|alphabet|, where
        L is the number of observations
    alphabet: tuple of the character alphabet, including the gap character
    merge_function: function used to map paths to label sequences, this could
        involve collapsing repeated labels or just removing gaps.
    '''
    def __init__(self, softmax, alphabet, merge_function=remove_gaps):
        self.softmax = softmax
        self.alphabet = alphabet
        self.merge_function = merge_function

        self.label_prob_ = dict()
        self.path_prob = dict()

        total_path_prob  = 0

        for path in itertools.product(range(len(alphabet)),repeat=len(self.softmax)):
            path_prob_ = np.product(self.softmax[np.arange(len(self.softmax)),np.array(path)])
            total_path_prob += path_prob_
            self.path_prob[path] = path_prob_

            label = self.merge_function([self.alphabet[l] for l in path])
            if label in self.label_prob_:
                self.label_prob_[label] += path_prob_
            else:
                self.label_prob_[label] = path_prob_

        assert(np.isclose(total_path_prob, 1.0))

        self.label_prob_ = OrderedDict(sorted(self.label_prob_.items(), key=operator.itemgetter(1), reverse=True))

    def top_label(self, n=1):
        l = list(self.label_prob_.items())
        if n == 1:
            return(l[0])
        elif n > len(l):
            return(l)
        else:
            return(l[:n])

    def label_prob(self, label):
        return(self.label_prob_.get(label,0.))

    def all_labels(self):
        return(self.label_prob_.keys())

    def prefix_prob(self, prefix):
        prefix_prob_ = 0
        for t in range(len(self.softmax)):
            for path in itertools.product(range(len(self.alphabet)),repeat=(t+1)):
                path_prob_ = np.product(self.softmax[np.arange(t+1),np.array(path)])
                label = self.merge_function([self.alphabet[l] for l in path])
                if label == prefix:
                    if path[-1] != 2:
                        prefix_prob_ += path_prob_
        return(prefix_prob_)

class joint_profile:
    def __init__(self, prof1, prof2):
        self.joint_label_prob_ = dict()
        self.prob_agree = 0.
        for label in prof1.all_labels():
            joint_prob = prof1.label_prob(label)*prof2.label_prob(label)
            self.joint_label_prob_[label] = joint_prob
            self.prob_agree += joint_prob
        self.joint_label_prob_ = OrderedDict(sorted(self.joint_label_prob_.items(), key=operator.itemgetter(1), reverse=True))

    def top_label(self, n=1):
        l = list(self.joint_label_prob_.items())
        if n == 1:
            return(l[0])
        elif n > len(l):
            return(l)
        else:
            return(l[:n])

    def label_prob(self, label):
        return(self.joint_label_prob_.get(label,0.))

def test_pair_forward(y1,y2, examples,envelope=None,forward_algorithm=consensus.pair_forward):
    alphabet = ('A','B','')
    alphabet_dict = {'A':0,'B':1,'':2}

    profile1=profile(y1,alphabet,remove_gaps)
    profile2=profile(y2,alphabet,remove_gaps)
    joint_prof = joint_profile(profile1, profile2)

    for label in examples:
        label_int = [alphabet_dict[i] for i in label]
        alpha,_,_  = forward_algorithm(label_int,y1,y2,mask=envelope)
        assert(np.isclose(consensus.pair_label_prob(alpha), joint_prof.label_prob(label)))

def test_prefix_search(y1,y2,envelope=None):
    alphabet = ('A','B','')
    toy_alphabet = OrderedDict([('A',0),('B',1)])

    profile1= profile(y1,alphabet,remove_gaps)
    profile2=profile(y2,alphabet,remove_gaps)
    joint_prof = joint_profile(profile1, profile2)

    top_label = joint_prof.top_label()
    search_top_label = consensus.pair_prefix_search_vec(y1,y2,alphabet=toy_alphabet)
    assert((top_label[0] == search_top_label[0]) and np.isclose(top_label[1] / joint_prof.prob_agree, search_top_label[1]))

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
