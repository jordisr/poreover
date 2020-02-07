import itertools
import operator
import os
import sys
import numpy as np
from collections import OrderedDict

import poreover.decoding as decoding

class profile:
    '''
    Simple class for probabilistic profiles. Label probabilities are calculated
    by enumerating all paths. Useful for unit testing of decoding on toy examples.

    Arguments:
    sofmax: numpy array of softmax probabilities, shape is Lx|alphabet|, where
        L is the number of observations
    alphabet: tuple of the character/gap alphabet
    merge_function: function used to map paths to label sequences, this could
        involve collapsing repeated labels or removing gaps.
    '''
    def __init__(self, softmax, alphabet, merge_function):
        self.softmax = softmax
        self.alphabet = alphabet
        self.merge_function = merge_function
        self.total_path_prob = 0
        self.label_prob_ = dict()
        self.path_prob = dict()

    def top_label(self, n=1):
        l = list(self.label_prob_.items())
        if n == 1:
            return(l[0])
        elif n > len(l):
            return(l)
        else:
            return(l[:n])

    def top_path(self, n=1):
        sorted_path_prob = sorted(self.path_prob.items(), key=operator.itemgetter(1), reverse=True)
        if n == 1:
            return(sorted_path_prob[0])
        elif n > len(l):
            return(sorted_path_prob)
        else:
            return(sorted_path_prob[:n])

    def viterbi_decode(self):
        top_path = self.top_path()[0]
        return self.merge_function([self.alphabet[l] for l in top_path])

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

class poreover_profile(profile):
    def __init__(self, prob, alphabet):
        super().__init__(prob, alphabet, decoding.remove_gaps)

        for path in itertools.product(range(len(alphabet)),repeat=len(self.softmax)):
            path_prob_ = np.product(self.softmax[np.arange(len(self.softmax)),np.array(path)])
            self.total_path_prob += path_prob_
            self.path_prob[path] = path_prob_

            label = self.merge_function([self.alphabet[l] for l in path])
            if label in self.label_prob_:
                self.label_prob_[label] += path_prob_
            else:
                self.label_prob_[label] = path_prob_

        assert(np.isclose(self.total_path_prob, 1.0))

        self.label_prob_ = OrderedDict(sorted(self.label_prob_.items(), key=operator.itemgetter(1), reverse=True))

def flipflop_transition(flipflop_size):
    a = np.ones((flipflop_size, flipflop_size))
    b = np.identity(flipflop_size)
    return np.block([[a,b],[a,b]])

class flipflop_profile(profile):
    def __init__(self, prob, alphabet):
        super().__init__(prob, alphabet, lambda x: decoding.transducer.remove_repeated(x).upper())
        self.flipflop_size = int(len(alphabet)/2)
        self.transition = flipflop_transition(self.flipflop_size)

        def extend_path(p):
            return([p[:]+[j] for j in np.where(self.transition[p[-1]] == 1)[0]])

        paths = [[i] for i in range(len(alphabet))]

        for t in range(1,len(self.softmax)):
            paths_ = []
            for i in range(len(paths)):
                for j in extend_path(paths[i]):
                    paths_.append(j)
            paths = paths_[:]

        for path in paths:
            path_prob_ = np.product(self.softmax[np.arange(len(self.softmax)),path])
            self.total_path_prob += path_prob_
            path_string = ''.join(np.take(self.alphabet, path))
            self.path_prob[path_string] = path_prob_

            label = self.merge_function(path_string)
            if label in self.label_prob_:
                self.label_prob_[label] += path_prob_
            else:
                self.label_prob_[label] = path_prob_

        self.label_prob_ = OrderedDict(sorted(self.label_prob_.items(), key=operator.itemgetter(1), reverse=True))

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
