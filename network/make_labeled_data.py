'''
Generate labeled training/testing data from nanoraw genome_resquiggle output
'''

import numpy as np
import h5py
from multiprocessing import Pool
import argparse, random, sys, glob, os, re

parser = argparse.ArgumentParser(description='Make training data from resquiggled reads')
parser.add_argument('--input', help='Location of nanoraw-processed FAST5 file/directory', required=True)
parser.add_argument('--output', default='nanoraw', help='Prefix for output files')
parser.add_argument('--unroll', type=int, default=100, help='Break reads into fixed-width segments')
parser.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale', 'none'], help='Type of normalization')
parser.add_argument('--threads', type=int, default=1, help='Processes to use')
parser.add_argument('--expand', default=False,action='store_true',help='Output one base per signal')
args = parser.parse_args()

# open files for output
#signal_file = open(args.output+'.signal','w')
#bases_file = open(args.output+'.bases','w')

def read_to_training(read_path):
    hdf = h5py.File(read_path,'r')

    read_path_base = ''.join(read_path.split('.')[:-1])

    # basic parameters
    read_string = list(hdf['/Raw/Reads'].keys())[0]
    read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
    read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
    read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

    # events (not used)
    #raw_events_path = '/Analyses/EventDetection_000/Reads/'+read_string+'/Events'
    #raw_events = hdf[raw_events_path]

    # raw signal
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
        signal_file = open(read_path_base+'.signal','w')
        bases_file = open(read_path_base+'.bases','w')
        print(os.path.basename(read_path))
        nanoraw_events = hdf[nanoraw_path]
        nanoraw_relative_start = hdf[nanoraw_path].attrs['read_start_rel_to_raw']
        #print(read_string, len(raw_signal))

        base_string = ''
        # create one base per signal for labels
        for norm_mean, norm_stdev, start, length, base in nanoraw_events:
            absolute_start = nanoraw_relative_start + start
            absolute_end = absolute_start + length
            #print(norm_mean - np.mean(raw_signal[absolute_start:absolute_end]))
            if args.expand:
                base_string += base.decode('UTF-8')*length
            else:
                base_string += (base.decode('UTF-8')+'-'*(length-1))

        raw_signal = raw_signal[nanoraw_relative_start:nanoraw_relative_start+start+length]

        # rescale signal
        if args.scaling == 'standard':
            # standardize
            norm_signal = (raw_signal - np.mean(raw_signal))/np.std(raw_signal)
        elif args.scaling == 'current':
            # convert to current
            norm_signal = (raw_signal+offset)/alpha # convert to pA
        elif args.scaling == 'median':
            # divide by median
            norm_signal = raw_signal / np.median(raw_signal)
        elif args.scaling == 'rescale':
            norm_signal = (raw_signal - np.mean(raw_signal))/(np.max(raw_signal) - np.min(raw_signal))
        elif args.scaling == 'none':
            norm_signal = raw_signal

        assert(len(norm_signal) == len(base_string))

        i = 0
        UNROLL = args.unroll
        while i + UNROLL < len(norm_signal):
            signal_output = ' '.join(map(str,norm_signal[i:i+UNROLL]))+'\n'
            base_output = ' '.join([b for b in base_string[i:i+UNROLL] if b != '-'])+'\n'

            # only write if there are bases to output
            if len(base_output) > 1:
                signal_file.write(signal_output)
                bases_file.write(base_output)
            i += UNROLL
    else:
        print('# read has not be resquiggled:', os.path.basename(read_path))

if __name__ == '__main__':

    path = args.input

    # for future parallel processing of directories
    #NUM_THREADS = 1
    NUM_THREADS = args.threads

    # summarize options
    print('# input:',args.input)
    print('# output:',args.output)
    print('# unroll:',args.unroll)
    print('# expand:',args.expand)
    print('# scaling:',args.scaling)

    if os.path.isdir(path):
        fast5_files = glob.glob(path+'/*.fast5')
        pool = Pool(processes=NUM_THREADS)
        pool.map(read_to_training, fast5_files)
    else:
        read_to_training(path)
