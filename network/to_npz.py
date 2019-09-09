# package training data into single NumPy binary npz format

import sys, os
import numpy as np

alphabet = {'A':0, 'C':1, 'G':2, 'T':3}

input_path = sys.argv[1]

signal_npy = np.loadtxt(input_path+'.signal', dtype=np.float32, delimiter=' ')

bases = np.loadtxt(input_path+'.bases', delimiter='\n',dtype='str')
bases_list = list(map(lambda x: np.array([alphabet[i] for i in x.split()]), bases))

seq_length = np.array(list(map(len,bases_list)))

bases_npy = np.concatenate(bases_list)

np.savez('training', signal=signal_npy, labels=bases_npy, row_lengths=seq_length)
