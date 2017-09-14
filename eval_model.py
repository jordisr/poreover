'''
Output test/training accuracy for model
'''
import numpy as np
import tensorflow as tf
import argparse
import glob, os

# some custom helper functions
import kmer
import batch

def accuracy(data_size, data_predict, data_label):
    total_kmers = 0
    total_matches = 0
    for size, predict, label in zip(data_size, data_predict, data_label):
        total_kmers += size
        total_matches += np.sum(predict[:size] == label[:size])
    return(total_matches/total_kmers)

# parse command line arguments
parser = argparse.ArgumentParser(description='Run the basecaller')

# data
parser.add_argument('--train_data', help='Location of training data', required=True)
parser.add_argument('--test_data', help='Location of test data (for test accuracy)')

# model
parser.add_argument('--latest', help='Load latest model in a directory', default='./')
parser.add_argument('--model', default=False, help='Saved model to run')
parser.add_argument('--all', default=False, help='Evaluate all models in a directory')

# options
parser.add_argument('--sample_size', type=float, default=0.01, help='Fraction of points to sample for accuracy')
parser.add_argument('--samples', type=int, default=1, help='Number of samples for accuracy')
args = parser.parse_args()

INPUT_DIM = 2

# load training data into memory (small files so this is OK for now)
#(train_events, train_bases) = batch.load_data(args.train_data)
(padded_train_data, padded_train_labels) = batch.load_data(args.train_data, INPUT_DIM)

# package in dataset iterator
train_dataset = batch.data_helper(padded_train_data, padded_train_labels, small_batch=False, return_length=True)
train_sizes = train_dataset.sequence_length

# for outputing test accuracy
if args.test_data:
    #(test_events, test_bases) = batch.load_data(args.test_data)
    (padded_test_data, padded_test_labels) = batch.load_data(args.test_data, INPUT_DIM)

    test_dataset = batch.data_helper(padded_test_data, padded_test_labels, small_batch=False, return_length=True)
    test_sizes = test_dataset.sequence_length

with tf.Session() as sess:

    # load model from checkpoint
    if args.model:
        model_list = [args.model]
    elif args.all:
        model_list = [os.path.splitext(g)[0] for g in sorted(glob.glob(args.all+'/*.index'))]
    else:
        model_list = [tf.train.latest_checkpoint(args.latest)]

    for model_file in model_list:

        saver = tf.train.import_meta_graph(model_file+'.meta') # loads latest model
        saver.restore(sess,model_file)
        graph = tf.get_default_graph()

        # load tensors needed for inference
        prediction = graph.get_tensor_by_name('prediction:0')
        X=graph.get_tensor_by_name('X:0')
        sequence_length=graph.get_tensor_by_name('sequence_length:0')

        test_accuracy = []
        train_accuracy = []

        for i in range(args.samples):

            if args.test_data:
                test_subset = np.random.choice(np.arange(len(padded_test_data)), int(len(padded_test_data)*args.sample_size))
                test_data_subset = np.take(padded_test_data, test_subset, axis=0)
                test_sizes_subset = np.take(test_sizes, test_subset, axis=0)
                test_labels_subset = np.take(padded_test_labels, test_subset, axis=0)

                predict_test = sess.run(prediction, feed_dict={X:test_data_subset, sequence_length:test_sizes_subset})
                test_accuracy.append(accuracy(test_sizes_subset, predict_test, test_labels_subset))

            train_subset = np.random.choice(np.arange(len(padded_train_data)), int(len(padded_train_data)*args.sample_size))
            train_data_subset = np.take(padded_train_data, train_subset, axis=0)
            train_sizes_subset = np.take(train_sizes, train_subset, axis=0)
            train_labels_subset = np.take(padded_train_labels, train_subset, axis=0)

            predict_train = sess.run(prediction, feed_dict={X:train_data_subset, sequence_length:train_sizes_subset})
            train_accuracy.append(accuracy(train_sizes_subset, predict_train, train_labels_subset))

        if args.test_data:
            print('model:',model_file,'samples:', args.samples,
            'train_accuracy_mean:',np.mean(train_accuracy),
            'train_accuracy_stdv:',np.std(train_accuracy),
            'test_accuracy_mean:',np.mean(test_accuracy),
            'test_accuracy_stdv:',np.std(test_accuracy))
        else:
            print('model:',model_file,'sample:',arg.samples,'train_accuracy:',np.mean(train_accuracy))
