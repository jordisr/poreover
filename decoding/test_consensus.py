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

class TestConsensusDecoding(unittest.TestCase):

    def test_something(self):
        self.assertTrue(1)

    def test_something_else(self):
        self.assertTrue(1)

if __name__ == '__main__':
    unittest.main()
