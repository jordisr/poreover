'''
TODO
- support for directory of FAST5 files and flexibility in FAST5 file naming
'''

import numpy as np
import tensorflow as tf
import h5py
import os, sys, re, argparse

# some custom helper functions
import kmer
import batch

# parse command line arguments
parser = argparse.ArgumentParser(description='Run the basecaller')
parser.add_argument('--model', help='Saved model to run')
parser.add_argument('--model_dir', help='Directory of models (loads latest)', default='./')
parser.add_argument('--events', help='File with events (.events)')
parser.add_argument('--fast5', default=False, help='FAST5 file to basecall (directories not currently supported)')
parser.add_argument('--stitch', action='store_true', default=False, help='Stitch list of kmers into one sequence')
parser.add_argument('--fasta', action='store_true', default=False, help='Write stitched output sequence in FASTA')
args = parser.parse_args()

if args.fast5:
    if os.path.isdir(args.fast5):
        sys.exit("FAST5 directories are not supported yet!")
    else:
        # parse FAST5 file, for now assume full file name scheme
        full_path = args.fast5
        file_name = full_path.split('/')[-1]
        path_match = re.match(r'.+_ch(\d+)_read(\d+)_strand\d?\.fast5', file_name)
        (channel, read) = (path_match.group(1), path_match.group(2))

        hdf = h5py.File(full_path,'r')
        #basecall_events_path = '/Analyses/Basecall_1D_000/BaseCalled_template/Events'
        raw_events_path = '/Analyses/EventDetection_000/Reads/Read_'+read+'/Events'
        #raw_signal_path = '/Raw/Reads/Read_'+read+'/Signal'

        #if basecall_events_path in hdf: basecall_events = hdf[basecall_events_path]
        if raw_events_path in hdf: raw_events = hdf[raw_events_path]['mean']
        #if raw_signal_path in hdf: raw_signal = hdf[raw_signal_pat

        # pad input sequences
        (padded_X,sizes) = (np.array([raw_events]), [len(raw_events)])
        padded_X = np.expand_dims(padded_X,axis=2)

elif args.events:
    # parse .events file
    raw_events = []
    with open(args.events, 'r') as f:
        for line in f:
            if len(line.split()) > 1:
                raw_events.append(np.array(list(map(lambda x: float(x),line.split()))))
    # pad input sequences
    (padded_X,sizes) = batch.pad(raw_events)
    padded_X = np.expand_dims(padded_X,axis=2)

else:
    sys.exit("An input file must be specified with --events or --fast5!")

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
        kmers = list(map(kmer.label2kmer, prediction[:length]))
        seq_counter += 1
        if args.stitch or args.fasta:
            sequence = kmer.stitch_kmers(kmers)
            if args.fasta:
                print(kmer.fasta_format('sequence '+str(seq_counter),sequence))
            else:
                print(sequence)
        else:
            print(kmers)
