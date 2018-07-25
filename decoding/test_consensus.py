'''
Tests to write/modify:

Original implementations:
consensus.pair_gamma
consensus.pair_forward
consensus.pair_label_prob
consensus.pair_prefix_prob
consensus.pair_prefix_search

Vectorized implementations:
consensus.forward_vec
consensus.alpha_ast_1d
consensus.prefix_prob_vec
consensus.prefix_prob_vec
consensus.pair_prefix_search_vec
consensus.prefix_search_vec
'''

import unittest
import numpy as np
from collections import OrderedDict

from testing import profile, joint_profile
from consensus import forward, pair_prefix_search, pair_gamma, prefix_search

'''
class TestDecoding(unittest.TestCase):

    def test_pair_label_prob(self):
        alphabet = ('A','B','')
        alphabet_dict = {'A':0,'B':1,'':2}

        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
        examples = ['AAAA','ABBA','ABA','AA','BB','A','B']

        profile1=profile(y1,alphabet)
        profile2=profile(y2,alphabet)
        joint_prof = joint_profile(profile1, profile2)

        for label in examples:
            label_int = [alphabet_dict[i] for i in label]
            alpha,_,_  = pair_forward(label_int,y1,y2)
            self.assertTrue(np.isclose(pair_label_prob(alpha), joint_prof.label_prob(label)))
'''

class TestForward(unittest.TestCase):

    def test_forward(self):
        alphabet = ('A','B','')
        alphabet_dict = {'A':0,'B':1,'':2}

        #y = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        #examples = ['AAAA','ABBA','ABA','AA','BB','A','B']
        y = np.array([[0.2,0.1,0.7],[0.2,0.3,0.5],[0.4,0.3,0.3],[0.5,0.15,0.35]])
        examples = ['ABA']
        prof=profile(y,alphabet)

        for label in examples:
            label_int = [alphabet_dict[i] for i in label]
            alpha  = forward(label_int, y)
            print(alpha)
            print(label)
            print(prof.label_prob(label))
            self.assertTrue(np.isclose(alpha[-1,-1], prof.label_prob(label)))

class TestVectorizedDecoding(unittest.TestCase):

    def test_pair_prefix_search(self):
        def helper(y1,y2):
            alphabet = ('A','B','')
            toy_alphabet = OrderedDict([('A',0),('B',1)])

            profile1= profile(y1,alphabet)
            profile2=profile(y2,alphabet)
            joint_prof = joint_profile(profile1, profile2)

            top_label = joint_prof.top_label()
            search_top_label = pair_prefix_search(y1,y2,alphabet=toy_alphabet)
            return((top_label[0] == search_top_label[0]) and np.isclose(top_label[1] / joint_prof.prob_agree, search_top_label[1]))

        y1 = y2 = np.array([[0.1,0.6,0.3],[0.4,0.2,0.4],[0.4,0.3,0.3],[0.2,0.8,0]])
        self.assertTrue(helper(y1,y2))

        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
        self.assertTrue(helper(y1,y2))

        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5]])
        self.assertTrue(helper(y1,y2))

        y1 = y2 = np.array([[0,0,1],[1,0,0],[0,1,0]])
        self.assertTrue(helper(y1,y2))

    def test_pair_gamma(self):
        def helper(y1,y2):
            alphabet = ('A','B','')
            toy_alphabet = OrderedDict([('A',0),('B',1)])
            profile1= profile(y1,alphabet)
            profile2=profile(y2,alphabet)
            joint_prof = joint_profile(profile1, profile2)

            gamma = pair_gamma(y1,y2)
            self.assertTrue(np.isclose(gamma[0,0], joint_prof.prob_agree))

        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
        helper(y1,y2)

        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5]])
        helper(y1,y2)

        y1 = y2 = np.array([[0,0,1],[1,0,0],[0,1,0]])
        helper(y1,y2)

class TestVectorized1D(unittest.TestCase):

    def test_prefix_search_vec(self):

        def helper(y):
            alphabet = ('A','B','')
            toy_alphabet = OrderedDict([('A',0),('B',1)])
            prof = profile(y,alphabet)
            top_label = prof.top_label()
            search_top_label = prefix_search(y,alphabet=toy_alphabet)
            return((top_label[0] == search_top_label[0]) and np.isclose(top_label[1], search_top_label[1]))

        y = np.array([[0.1,0.6,0.3],[0.4,0.2,0.4],[0.4,0.3,0.3],[0.2,0.8,0]])
        self.assertTrue(helper(y))

        y = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
        self.assertTrue(helper(y))

        y = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5]])
        self.assertTrue(helper(y))

if __name__ == '__main__':
    unittest.main()
