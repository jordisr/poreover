import unittest
import numpy as np
import os
from collections import OrderedDict
from testing import poreover, poreover_profile, flipflop_profile, joint_profile
import poreover.decoding as decoding

class beam_1d_toy(unittest.TestCase):
    '''
    1D decoding of toy data
    '''
    def test(self):
        y = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        prof = poreover_profile(y,('A','B',''))
        result = decoding.decoding_cpp.cpp_beam_search(np.log(y), alphabet_="AB")
        self.assertTrue(result == prof.top_label()[0])

    def test2(self):
        y = np.array([[0.4,0.5,0.1],[0.4,0.2,0.4],[0.3,0.5,0.2]])
        prof = poreover_profile(y,('A','B',''))
        result = decoding.decoding_cpp.cpp_beam_search(np.log(y), alphabet_="AB")
        print(prof.top_label()[0], "------", result)
        self.assertTrue(result == prof.top_label()[0])

class beam_2d_toy(unittest.TestCase):
    '''
    2D decoding of toy data
    '''
    def test_same(self):
        y = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        result_1d = decoding.decoding_cpp.cpp_beam_search(np.log(y), alphabet_="AB")
        result_2d = decoding.decoding_cpp.cpp_beam_search_2d(np.log(y), np.log(y), alphabet_="AB")
        self.assertTrue(result_1d == result_2d)

    #def test_same_2(self):
    #    y = np.array([[0.4,0.5,0.1],[0.4,0.2,0.4],[0.3,0.5,0.2]])
    #    result_1d = decoding.decoding_cpp.cpp_beam_search(np.log(y), alphabet_="AB")
    #    result_2d = decoding.decoding_cpp.cpp_beam_search_2d(np.log(y), np.log(y), alphabet_="AB", beam_width_=2, method="grid")
    #    print(result_1d, result_2d)
    #    self.assertTrue(result_1d == result_2d)

    def test_full_envelope(self):
        y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
        y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
        full_seq = decoding.decoding_cpp.cpp_beam_search_2d(np.log(y1), np.log(y2), alphabet_="AB")
        prof1 = poreover_profile(y1,('A','B',''))
        prof2 = poreover_profile(y2, ('A','B',''))
        joint_prof = joint_profile(prof1, prof2)
        self.assertTrue(full_seq == joint_prof.top_label()[0])

    def test_flipflop_same(self):
        alphabet_tuple = ('A','B','a','b')
        alphabet = "AB"

        y = np.array([[0.8,0.1,0.05,0.05],[0.1,0.3,0.5,0.1],[0.7,0.2,0.05,0.05],[0.1,0.1,0.2,0.6]], dtype=np.float32)

        prof=flipflop_profile(y, alphabet_tuple)
        joint_prof=joint_profile(prof, prof)
        result_1d = decoding.decoding_cpp.cpp_beam_search(np.log(y), alphabet_="AB", model_='ctc_flipflop')
        result_2d = decoding.decoding_cpp.cpp_beam_search_2d(np.log(y), np.log(y), alphabet_="AB", method_="row", model_='ctc_flipflop')
        self.assertTrue(result_1d == result_2d)

class beam_2d_same(unittest.TestCase):
    '''
    Decode read with itself
    '''
    def setUp(self):
        self.model = decoding.decode.model_from_trace(os.path.abspath(os.path.dirname(__file__))+"/poreover.csv")
        self.t_max = self.model.log_prob.shape[0]

    def test_same(self):
        y = self.model.log_prob
        result_1d = decoding.decoding_cpp.cpp_beam_search(y, beam_width_=10)
        result_2d = decoding.decoding_cpp.cpp_beam_search_2d(y, y, beam_width_=10)
        print(result_1d, result_2d)
        self.assertTrue(result_1d == result_2d)

    def test_same_row_col(self):
        y = self.model.log_prob
        result_1d = decoding.decoding_cpp.cpp_beam_search(y, beam_width_=10)
        envelope_width = 10
        envelope_ranges = np.array([(max(0,i-envelope_width),min(i+envelope_width, self.t_max)) for i in range(self.t_max)])
        result_2d = decoding.decoding_cpp.cpp_beam_search_2d(y, y, envelope_ranges.tolist(), beam_width_=10, method_="row")
        print(result_1d, result_2d)
        self.assertTrue(result_1d == result_2d)

    #def test_same_2(self):
    #    y = self.model.log_prob
    #    result_1d = decoding.decoding_cpp.cpp_beam_search(y, alphabet_="ACGT", beam_width_=10)
    #    result_2d = decoding.decoding_cpp.cpp_beam_search_2d(y, y, alphabet_="ACGT", beam_width_=10, method_="grid")
    #    print(result_1d, result_2d)
    #    print(decoding.decoding_cpp.cpp_forward(y, result_1d), decoding.decoding_cpp.cpp_forward(y, result_2d))
    #    self.assertTrue(result_1d == result_2d)

    def test_full_envelope(self):
        full_seq = decoding.decoding_cpp.cpp_beam_search_2d(self.model.log_prob, self.model.log_prob)
        envelope_ranges = np.tile([0, self.t_max-1], (self.t_max, 1))
        full_envelope_seq = decoding.decoding_cpp.cpp_beam_search_2d(self.model.log_prob, self.model.log_prob, envelope_ranges.tolist())
        self.assertTrue(full_seq == full_envelope_seq)

    def test_diagonal_envelope(self):
        result_1d = decoding.decoding_cpp.cpp_beam_search(self.model.log_prob)
        envelope_ranges = np.array([(i,i+1) for i in range(self.t_max)])
        result_2d = decoding.decoding_cpp.cpp_beam_search_2d(self.model.log_prob, self.model.log_prob, envelope_ranges.tolist())
        self.assertTrue(result_1d == result_2d)

    #def test_diagonal_envelope(self):
    #    result_1d = decoding.decoding_cpp.cpp_beam_search(self.model.log_prob)
    #    envelope_ranges = np.array([(i,i+10) for i in range(self.t_max)])
    #    result_2d = decoding.decoding_cpp.cpp_beam_search_2d(self.model.log_prob, self.model.log_prob, envelope_ranges.tolist(), method_="grid")
    #    self.assertTrue(result_1d == result_2d)

if __name__ == '__main__':
    unittest.main()
