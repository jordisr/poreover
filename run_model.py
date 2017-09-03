'''
TODO
- read FAST5 file and pull out events
- support for directory of FAST5 files
- output in fixed-width FASTA format
'''

import numpy as np
import tensorflow as tf
import argparse

# some custom helper functions
import kmer
import batch

# parse command line arguments
parser = argparse.ArgumentParser(description='Run the basecaller')
parser.add_argument('--model', help='Saved model to run')
parser.add_argument('--model_dir', help='Directory of models (loads latest)', default='./')
parser.add_argument('--events', help='File with events (.events)', required=True)
#parser.add_argument('--fast5', help='FAST5 file or directory to basecall')
parser.add_argument('--stitch', action='store_true', default=False, help='Stitch list of kmers into one sequence')
#parser.add_argument('--fasta_output', action='store_true', default=False, help='Write stitched sequence in FASTA')
args = parser.parse_args()

# load training data to make a test prediction
raw_events = []
with open(args.events, 'r') as f:
     for line in f:
            if len(line.split()) > 1:
                raw_events.append(np.array(list(map(lambda x: float(x),line.split()))))

(padded_X,sizes) = batch.pad(raw_events)
padded_X = np.expand_dims(padded_X,axis=2)

with tf.Session() as sess:

    # load model from checkpoint
    if args.model is not None:
        model_file = args.model
    else:
        model_file = tf.train.latest_checkpoint(args.model_dir)
    saver = tf.train.import_meta_graph(model_file+'.meta') # loads latest model
    saver.restore(sess,model_file)
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    # load tensors needed for inference
    prediction = graph.get_tensor_by_name('prediction:0')
    X=graph.get_tensor_by_name('X:0')

    # make prediction
    predict_ = sess.run(prediction, feed_dict={X:padded_X})

    for prediction in predict_:
        kmers = list(map(kmer.label2kmer, prediction))
        if args.stitch:
            sequence = kmer.stitch_kmers(kmers)
            print(sequence)
            #if args.fasta_output:
            #    print(sequence)
            #else:
            #    print(sequence)
        else:
            print(kmers)
