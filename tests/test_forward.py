import unittest
import numpy as np
import sys
from collections import OrderedDict
from testing import decoding, poreover_profile, flipflop_profile, joint_profile

class TestForwardAlgorithm(unittest.TestCase):

    def test_fw_poreover(self):
        alphabet_tuple = ('A','B','')
        alphabet = "AB"

        y = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]], dtype=np.float32)
        examples = ['AAAA','ABBA','ABA','AAA','BBB', 'AA','BB','A','B']
        prof=poreover_profile(y, alphabet_tuple)

        for label in examples:
            fw_prob_actual =  np.log(prof.label_prob(label))
            fw_prob  = decoding.decoding_cpp.cpp_forward(np.log(y), label, alphabet)
            print(label, fw_prob_actual, fw_prob)
            self.assertTrue(np.isclose(fw_prob_actual, fw_prob))

    def test_fw_flipflop(self):
        alphabet_tuple = ('A','B','a','b')
        alphabet = "AB"

        y = np.array([[0.8,0.1,0.05,0.05],[0.1,0.3,0.5,0.1],[0.7,0.2,0.05,0.05],[0.1,0.1,0.2,0.6]], dtype=np.float32)
        examples = ['AAAA','ABBA','ABA','AAA','BBB', 'AA','BB','A','B']
        prof=flipflop_profile(y, alphabet_tuple)

        for label in examples:
            fw_prob_actual =  np.log(prof.label_prob(label))
            fw_prob  = decoding.decoding_cpp.cpp_forward(np.log(y), label, alphabet, model_='ctc_flipflop')
            print(label, fw_prob_actual, fw_prob)
            self.assertTrue(np.isclose(fw_prob_actual, fw_prob))

if __name__ == '__main__':
    unittest.main()
