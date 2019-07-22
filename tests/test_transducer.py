import unittest
import numpy as np
import os
from collections import OrderedDict
from testing import decoding, poreover_profile, flipflop_profile, joint_profile

class test_viterbi_decode(unittest.TestCase):
    def test_poreover(self):
        y = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        model = decoding.transducer.poreover(y, "AB")
        prof = poreover_profile(y,('A','B',''))
        self.assertTrue(model.viterbi_decode() == prof.viterbi_decode())

class test_viterbi_acceptor(unittest.TestCase):
    def test_cpp_with_best_path(self):
        '''
        Best path of Viterbi-decoded sequence should be global best (Viterbi) path
        '''
        transducer = decoding.decode.model_from_trace(os.path.abspath(os.path.dirname(__file__))+"/poreover.csv")
        viterbi_seq, viterbi_path = transducer.viterbi_decode(return_path=True)
        acceptor_path = decoding.decoding_cpp.cpp_viterbi_acceptor(transducer.log_prob.astype(np.float64), viterbi_seq)
        self.assertTrue(np.all(viterbi_path == acceptor_path))

    def test_cy_with_best_path(self):
        '''
        Best path of Viterbi-decoded sequence should be global best (Viterbi) path
        '''
        transducer = decoding.decode.model_from_trace(os.path.abspath(os.path.dirname(__file__))+"/poreover.csv")
        viterbi_seq, viterbi_path = transducer.viterbi_decode(return_path=True)
        acceptor_path = decoding.decoding_cy.viterbi_acceptor(transducer.log_prob.astype(np.float64), viterbi_seq)
        self.assertTrue(np.all(viterbi_path == acceptor_path))

if __name__ == '__main__':
    unittest.main()
