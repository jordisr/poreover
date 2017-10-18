'''
Generate raw signal training data from nanoraw genome_resquiggle output
'''

import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
import argparse, random, sys, glob, os

parser = argparse.ArgumentParser(description='Make training data from nanopolish')
parser.add_argument('--input', help='Location of nanoraw-processed FAST5 file/directory', required=True)
parser.add_argument('--output', default='nanoraw', help='Prefix for output files')
parser.add_argument('--unroll', type=int, default=100, help='Break reads into fixed-width segments')
parser.add_argument('--split', type=float, default=0.5, help='Create training/test validation split with X%% in test set')
parser.add_argument('--threads', type=int, default=1, help='Processes to use')
args = parser.parse_args()

# open filehandles for output
if args.split is not None:
    print("opening training/test split filehandles")
    train_events_file = open(args.output+'.train.signal','w')
    train_bases_file = open(args.output+'.train.bases','w')
    test_events_file = open(args.output+'.test.signal','w')
    test_bases_file = open(args.output+'.test.bases','w')
else:
    events_file = open(args.output+'.signal','w')
    bases_file = open(args.output+'.bases','w')

def read_to_training(read_path):
    hdf = h5py.File(read_path,'r')

    # basic parameters
    read_string = list(hdf['/Raw/Reads'].keys())[0]
    read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
    read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
    read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

    # raw events and signals
    raw_events_path = '/Analyses/EventDetection_000/Reads/'+read_string+'/Events'
    raw_events = hdf[raw_events_path]
    raw_signal_path = '/Raw/Reads/'+read_string+'/Signal'
    raw_signal = hdf[raw_signal_path]
    assert(len(raw_signal) == read_duration)

    # for converting raw signal to current (pA)
    alpha = hdf['UniqueGlobalKey']['channel_id'].attrs['digitisation'] / hdf['UniqueGlobalKey']['channel_id'].attrs['range']
    offset = hdf['UniqueGlobalKey']['channel_id'].attrs['offset']
    sampling_rate = hdf['UniqueGlobalKey']['channel_id'].attrs['sampling_rate']

    # nanoraw genome_resquiggle
    nanoraw_path = '/Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'
    if nanoraw_path in hdf:
        nanoraw_events = hdf[nanoraw_path]
        nanoraw_relative_start = hdf[nanoraw_path].attrs['read_start_rel_to_raw']
        #print(read_string, len(raw_signal))

        base_string = ''
        # create one base per signal for labels
        for norm_mean, norm_stdev, start, length, base in nanoraw_events:
            absolute_start = nanoraw_relative_start + start
            absolute_end = absolute_start + length
            #print(norm_mean - np.mean(raw_signal[absolute_start:absolute_end]))
            base_string += base.decode('UTF-8')*length

        # rescale signal based on range of nanoraw data (possibly other stuff in the future)
        raw_signal = raw_signal[nanoraw_relative_start:nanoraw_relative_start+start+length]
        norm_signal = raw_signal / np.median(raw_signal)
        #norm_signal = (raw_signal+offset)/alpha
        print(len(norm_signal), len(base_string))

        i = 0
        #output_data = []
        #output_labels = []

        UNROLL = args.unroll
        while i + UNROLL < len(norm_signal):
            #output_data.append ( norm_signal[i:i+UNROLL] )
            #output_labels.append( base_string[i:i+UNROLL] )
            if args.split is not None:
                if (random.random() > args.split):
                    test_events_file.write(' '.join(map(str,norm_signal[i:i+UNROLL]))+'\n')
                    test_bases_file.write(' '.join(base_string[i:i+UNROLL])+'\n')
                else:
                    train_events_file.write(' '.join(map(str,norm_signal[i:i+UNROLL]))+'\n')
                    train_bases_file.write(' '.join(base_string[i:i+UNROLL])+'\n')
            else:
                events_file.write(' '.join(map(str,norm_signal[i:i+UNROLL]))+'\n')
                bases_file.write(' '.join(base_string[i:i+UNROLL])+'\n')
            i += UNROLL

    else:
        print('Read has not be resquiggled', read_path)

if __name__ == '__main__':

    path = args.input

    # for parallel processing of directories
    NUM_THREADS = args.threads

    function_to_map = read_to_training

    if os.path.isdir(path):
        fast5_files = glob.glob(path+'/*.fast5')
        pool = Pool(processes=NUM_THREADS)
        pool.map(function_to_map, fast5_files)
    else:
        function_to_map(path)
