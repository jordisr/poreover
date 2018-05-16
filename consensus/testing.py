import itertools
import operator
import numpy as np

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
    def __init__(self,softmax, alphabet, merge_function):
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
