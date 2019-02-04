'''
Load saved model and basecall single read
'''
import numpy as np
import tensorflow as tf
import h5py
import os, sys, re, argparse, glob
import pickle

import decoding
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

def call(args):
    INPUT_DIM = 1 # raw signal
    WINDOW_SIZE = args.window

    if args.fast5:
        if os.path.isdir(args.fast5):
            fast5_files =  glob.glob(args.fast5+'/*.fast5')
            print("Found",len(fast5_files),"files to basecall")
            for fast5 in fast5_files:
                args.fast5 = fast5
                call(args)
            sys.exit()
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

        if args.logits:
            logits_ = sess.run(logits, feed_dict={X:stacked,sequence_length:sizes}).astype('float32')
            if args.logits == 'csv':
                np.savetxt(args.out+'.csv', np.concatenate(sess.run(tf.nn.softmax(logits_))), delimiter=',', header=','.join(['A','C','G','T','-']))
            else:
                np.save(args.out, logits_)

        if args.decoding == 'prefix':
            logits_ = sess.run(logits, feed_dict={X:stacked,sequence_length:sizes})
            softmax = sess.run(tf.nn.softmax(logits_))
            prediction_ = list()
            beam_search_counter = 0
            beam_search_total = len(softmax)

            from multiprocessing import Pool

            assert(len(softmax)==len(sizes))

            def basecall_segment(i):
                return(decoding.prefix_search(softmax[i][:sizes[i]])[0])

            NUM_THREADS = args.ctc_threads
            with Pool(processes=NUM_THREADS) as pool:
                basecalls = pool.map(basecall_segment, range(len(softmax)))

            sequence = ''.join(basecalls)

        elif args.decoding == 'greedy':
            logits_ = sess.run(logits, feed_dict={X:stacked,sequence_length:sizes})
            softmax = sess.run(tf.nn.softmax(logits_))

            assert(len(softmax)==len(sizes))

            def basecall_segment(i):
                return(decoding.greedy_search(softmax[i][:sizes[i]]))

            basecalls = [basecall_segment(i) for i in range(len(softmax))]
            sequence = ''.join(basecalls)

        elif args.decoding == 'beam':
            # make prediction
            prediction_ = sess.run(prediction, feed_dict={X:stacked,sequence_length:sizes})

            # stitch decoded sequence together
            sequence = ''
            for length_iter, pred_iter in zip(sizes,prediction_):
                sequence_segment = list(map(label2base, pred_iter[:length_iter]))
                sequence += ''.join(sequence_segment)

        # output decoded sequence
        if args.decoding != 'none':
            with open(args.out+'.fasta','a') as fasta_file:
                if args.fast5:
                    fasta_header = os.path.basename(args.fast5)
                elif args.signal:
                    fasta_header = os.path.basename(args.signal)
                print(fasta_format(fasta_header,sequence),file=fasta_file)
