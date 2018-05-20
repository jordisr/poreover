'''
TODO: support for directory of FAST5 files
'''
import numpy as np
import tensorflow as tf
import h5py
import os, sys, re, argparse
import pickle
import ctc

# some custom helper functions
import batch

def label2base(l):
    if l == 0:
        return 'A'
    elif l == 1:
        return 'C'
    elif l == 2:
        return 'G'
    elif l == 3:
        return 'T'
    elif l == 4:
        return ''

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

# parse command line arguments
parser = argparse.ArgumentParser(description='Run the basecaller')
parser.add_argument('--model', help='Saved model to load (if directory, loads latest from checkpoint file)', default='./models/r9')
parser.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale'], help='Type of preprocessing (should be same as training)')
parser.add_argument('--signal', help='File with space-delimited signal for testing')
parser.add_argument('--fast5', default=False, help='FAST5 file to basecall (directories not currently supported)')
parser.add_argument('--fasta', action='store_true', default=False, help='Write output sequence in FASTA')
parser.add_argument('--window', type=int, default=200, help='Call read using chunks of this size')
parser.add_argument('--logits', default=False, help='Pickle output logits to file')
parser.add_argument('--debug_ctc', default=False, action='store_true', help='Use own implementation of CTC decoding (WARNING: Does not collapse repeated characters)')
parser.add_argument('--ctc_threads', type=int, default=1, help='Number of threads to use for decoding')
parser.add_argument('--no_stack', default=False, action='store_true', help='Basecall [1xSIGNAL_LENGTH] tensor instead of splitting it into windows (slower)')
args = parser.parse_args()

INPUT_DIM = 1 # raw signal
WINDOW_SIZE = args.window

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

        raw_signal = [s for s in raw_signal if 200 < s < 800] # very rough heuristic for abasic region

        # rescale signal (should be same as option selected in make_labeled_data.py)
        if args.scaling == 'standard':
            # standardize
            signal = (raw_signal - np.mean(raw_signal))/np.std(raw_signal)
        elif args.scaling == 'current':
            # convert to current
            signal = (raw_signal+offset)/alpha # convert to pA
        elif args.scaling == 'median':
            # divide by median
            signal = raw_signal / np.median(raw_signal)
        elif args.scaling == 'rescale':
            signal = (raw_signal - np.mean(raw_signal))/(np.max(raw_signal) - np.min(raw_signal))

elif args.signal:
    raw_events = []
    with open(args.signal, 'r') as f:
        for line in f:
            if len(line.split()) > 1:
                raw_events.append(np.array(list(map(lambda x: float(x),line.split()))))
    # pad input sequences
    #sizes = [len(i) for i in raw_events]
    #padded_X = np.reshape(batch.pad(raw_events), (len(raw_events), -1, INPUT_DIM))
    signal = raw_events[0] # just take first line, don't normalize

else:
    sys.exit("An input file must be specified with --signal or --fast5!")

if not args.no_stack:
    # split signal into blocks to allow for faster basecalling with the GPU
    rounded = int(len(signal)/WINDOW_SIZE)*WINDOW_SIZE
    stacked = np.reshape(signal[:rounded], (-1, WINDOW_SIZE, INPUT_DIM))
    sizes = [len(i) for i in stacked]

    if rounded < len(signal):
        last_row = np.zeros(WINDOW_SIZE)
        last_row[:len(signal)-rounded] = signal[rounded:]
        last_row = np.expand_dims(last_row,1)
        sizes.append(len(signal)-rounded)
        stacked = np.vstack((stacked,np.expand_dims(last_row,0)))
else:
    stacked = np.reshape(np.expand_dims(signal,axis=1), (1, len(signal), INPUT_DIM))
    sizes = [len(i) for i in stacked]

with tf.Session() as sess:

    # if model argument is a directory load the latest model in it
    if os.path.isdir(args.model):
        model_file = tf.train.latest_checkpoint(args.model)
    else:
        model_file = args.model

    saver = tf.train.import_meta_graph(model_file+'.meta') # loads latest model
    saver.restore(sess,model_file)
    graph = tf.get_default_graph()

    # load tensors needed for inference
    prediction = graph.get_tensor_by_name('prediction:0')
    X=graph.get_tensor_by_name('X:0')
    sequence_length=graph.get_tensor_by_name('sequence_length:0')
    logits=graph.get_tensor_by_name('logits:0')

    if args.debug_ctc:
        logits_ = sess.run(logits, feed_dict={X:stacked,sequence_length:sizes})
        softmax = sess.run(tf.nn.softmax(logits_))
        prediction_ = list()
        beam_search_counter = 0
        beam_search_total = len(softmax)

        from multiprocessing import Pool

        assert(len(softmax)==len(sizes))

        def basecall_segment(i):
            return(ctc.prefix_search(softmax[i][:sizes[i]])[0])

        NUM_THREADS = args.ctc_threads
        with Pool(processes=NUM_THREADS) as pool:
            basecalls = pool.map(basecall_segment, range(len(softmax)))

        sequence = ''.join(basecalls)

        '''
        # version without multiprocessing
        for size_i,softmax_i in zip(sizes,softmax):
            prediction_.append(ctc.prefix_search(softmax_i[:size_i])[0])
            print('Prefix search done:',beam_search_counter,'of',beam_search_total, file=sys.stderr)
            beam_search_counter += 1

        # stitch decoded sequence together
        sequence = ''
        for s in prediction_:
            sequence += s
        '''

    else:
        # make prediction
        prediction_ = sess.run(prediction, feed_dict={X:stacked,sequence_length:sizes})

        # stitch decoded sequence together
        sequence = ''
        for length_iter, pred_iter in zip(sizes,prediction_):
            sequence_segment = list(map(label2base, pred_iter[:length_iter]))
            sequence += ''.join(sequence_segment)

    if args.logits:
        logits_ = sess.run(logits, feed_dict={X:stacked,sequence_length:sizes})
        #pickle.dump(logits_, open(args.logits,'wb'))
        #print(sess.run(tf.nn.softmax(logits_)).shape)
        np.array(sess.run(tf.nn.softmax(logits_))).astype('float32').tofile(args.logits)

    # output decoded sequence
    if args.fasta:
        if args.fast5:
            fasta_header = os.path.basename(args.fast5)
        elif args.signal:
            fasta_header = os.path.basename(args.signal)
        print(fasta_format(fasta_header,sequence))
    else:
        print(sequence)
