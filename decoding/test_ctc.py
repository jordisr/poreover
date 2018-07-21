'''
Tests to write/modify:

ctc.greedy_search
ctc.remove_gaps
ctc.forward
ctc.forward_add_column
ctc.forward_prefix_prob
ctc.prefix_search
'''

import unittest
import numpy as np
from collections import OrderedDict

from testing import profile
from ctc import remove_gaps, greedy_search, prefix_search

class TestUtils(unittest.TestCase):

    def test_remove_gaps(self):
        self.assertEqual(remove_gaps(['A','','B']), 'AB')
        self.assertEqual(remove_gaps(['A','-','B']), 'AB')
        self.assertEqual(remove_gaps(['-','A','A','-','','-','B']), 'AAB')
        self.assertEqual(remove_gaps('A-B'),'AB')

    def test_greedy_search(self):
        alphabet = ['A','B','-']

        y = np.array([[0.5,0.1,0.4],[0.5,0.1,0.4],[0.5,0.1,0.4],[0.,0.1,0.9]])
        self.assertEqual(greedy_search(y, alphabet),'AAA')

        y = np.array([[0.5,0.1,0.4]])
        self.assertEqual(greedy_search(y, alphabet),'A')

        y = np.array([[0,0,1]])
        self.assertEqual(greedy_search(y, alphabet),'')

        y = np.array([[0.5,0.5,0.]])
        self.assertEqual(greedy_search(y, alphabet),'A')

        y = np.array([[0.5,0.6,0.]])
        self.assertEqual(greedy_search(y, alphabet),'B')

class TestDecoding(unittest.TestCase):

    def test_prefix_search(self):

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

    def test_something_else(self):
        self.assertTrue(1)

if __name__ == '__main__':
    unittest.main()
