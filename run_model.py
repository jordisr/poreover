'''
TODO
- support for directory of FAST5 files and flexibility in FAST5 file naming
'''

import numpy as np
import tensorflow as tf
import h5py
import os, sys, re, argparse

# some custom helper functions
import batch

def label2base(l):
    if l == 1:
        return 'A'
    elif l == 2:
        return 'C'
    elif l == 3:
        return 'G'
    elif l == 4:
        return 'G'

def fasta_format(name, seq, width=80):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

# parse command line arguments
parser = argparse.ArgumentParser(description='Run the basecaller')
parser.add_argument('--model', help='Saved model to run')
parser.add_argument('--model_dir', help='Directory of models (loads latest)', default='./')
parser.add_argument('--signal', help='File with raw signal (.signal)')
parser.add_argument('--fast5', default=False, help='FAST5 file to basecall (directories not currently supported)')
parser.add_argument('--fasta', action='store_true', default=False, help='Write output sequence in FASTA')
args = parser.parse_args()

ALPHABET = 'ACGT'
INPUT_DIM = 1 # raw signal

if args.fast5:
    if os.path.isdir(args.fast5):
        sys.exit("FAST5 directories are not supported yet!")
    else:
        hdf = h5py.File(args.fast5,'r')

        # basic parameters
        read_string = list(hdf['/Raw/Reads'].keys())[0]
        read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
        read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
        read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

        # raw events and signals
        raw_signal_path = '/Raw/Reads/'+read_string+'/Signal'
        raw_signal = hdf[raw_signal_path]
        assert(len(raw_signal) == read_duration)

        # for converting raw signal to current (pA)
        alpha = hdf['UniqueGlobalKey']['channel_id'].attrs['digitisation'] / hdf['UniqueGlobalKey']['channel_id'].attrs['range']
        offset = hdf['UniqueGlobalKey']['channel_id'].attrs['offset']
        sampling_rate = hdf['UniqueGlobalKey']['channel_id'].attrs['sampling_rate']

        # rescale signal (currently no normalization or detection of abasic region)
        norm_signal = (raw_signal+offset)/alpha
        sizes = [len(norm_signal)]
        padded_X = np.array([np.expand_dims(norm_signal,axis=1)])

elif args.signal:
    # parse .events file
    raw_events = []
    with open(args.signal, 'r') as f:
        for line in f:
            if len(line.split()) > 1:
                raw_events.append(np.array(list(map(lambda x: float(x),line.split()))))
    # pad input sequences
    sizes = [len(i) for i in raw_events]
    padded_X = np.reshape(batch.pad(raw_events), (len(raw_events), -1, INPUT_DIM))

else:
    sys.exit("An input file must be specified with --signal or --fast5!")

with tf.Session() as sess:

    # load model from checkpoint
    if args.model is not None:
        model_file = args.model
    else:
        model_file = tf.train.latest_checkpoint(args.model_dir)
    saver = tf.train.import_meta_graph(model_file+'.meta') # loads latest model
    saver.restore(sess,model_file)
    graph = tf.get_default_graph()

    # load tensors needed for inference
    prediction = graph.get_tensor_by_name('prediction:0')
    X=graph.get_tensor_by_name('X:0')
    sequence_length=graph.get_tensor_by_name('sequence_length:0')

    # make prediction
    predict_ = sess.run(prediction, feed_dict={X:padded_X,sequence_length:sizes})
    seq_counter = 0
    for length, prediction in zip(sizes,predict_):
        kmers = list(map(label2base, prediction[:length]))
        sequence = ''.join(kmers) # provisional
        seq_counter += 1
        if args.fasta:
            print(fasta_format('sequence '+str(seq_counter),sequence))
        else:
            print(sequence)
