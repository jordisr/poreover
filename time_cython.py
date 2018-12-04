# Minimal example to time Cython decoding
import sys
import timeit

setup='''
import numpy as np
poreover_path = '/Users/jordi/work/poreover/'
sys.path.append(poreover_path)
import decoding
import cy
from pair_align_decode import load_logits
logits1 = load_logits(poreover_path+'data/read1.npy').astype(np.float64)
logits2 = load_logits(poreover_path+'data/read2.npy',reverse_complement=True).astype(np.float64)
'''
n=5
print('Testing partial Cython implementation')
#time = timeit.timeit('[cy.decoding.prefix_search_log(logits1[i]) for i in range(155)]', number=n, setup=setup)/n
time = timeit.timeit('[cy.decoding.pair_prefix_search_log(logits1[i], logits2[i]) for i in range(10)]', number=n, setup=setup)
print('\tAverage of {}s over {} iterations'.format(time, n))
print('Testing full Cython implementation')
time = timeit.timeit('[cy.decoding2.pair_prefix_search_log(logits1[i], logits2[i]) for i in range(10)]', number=n, setup=setup)
#time = timeit.timeit('[cy.decoding2.prefix_search_log(logits1[i]) for i in range(155)]', number=n, setup=setup)/n
print('\tAverage of {}s over {} iterations'.format(time, n))
