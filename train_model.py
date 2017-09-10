'''
TODO
- double check loss expression
- load model to resume training
- add additional hidden layers
- move model initialization and architecture to separate class for modularity
'''

import numpy as np
import tensorflow as tf
import argparse

# some custom helper functions
import batch
import kmer

def accuracy(data_size, data_predict, data_label):
    total_kmers = 0
    total_matches = 0
    for size, predict, label in zip(data_size, data_predict, data_label):
        total_kmers += size
        total_matches += np.sum(predict[:size] == label[:size])
    return(total_matches/total_kmers)

# parse command line arguments
parser = argparse.ArgumentParser(description='Train the basecaller')
parser.add_argument('--data', help='Location of training data', required=True)
parser.add_argument('--test_data', help='Location of test data (for test accuracy)')
parser.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
#parser.add_argument('--resume_from', help='Resume training from checkpoint file')

parser.add_argument('--training_steps', type=int, default=1000, help='Number of iterations to run training (default: 1000)')
parser.add_argument('--save_every', type=int, default=10000, help='Frequency with which to save checkpoint files (default: 10000)')
parser.add_argument('--accuracy_every', type=int, default=1000, help='Frequency with which to output test/training accuracy')
parser.add_argument('--loss_every', type=int, default=1000, help='Frequency with which to output loss')
#parser.add_argument('--loss_file',type=int,help='Write loss to a file instead of stdout')
args = parser.parse_args()

# model parameters
BATCH_SIZE = 32 # number of read fragments to use at a time
NUM_NEURONS = 100 # how many neurons
NUM_LAYERS = 1 # NOT CURRENTLY USED
NUM_OUTPUTS = 4096 #number of possible 6-mers

# user options
TRAINING_STEPS = args.training_steps #number of iterations of SGD
CHECKPOINT_ITER = args.save_every
LOSS_ITER = args.loss_every # how often to output loss
ACCURACY_ITER = args.accuracy_every

# load training data into memory (small files so this is OK for now)
(train_events, train_bases) = batch.load_data(args.data)

# pad data and labels
(padded_X,sizes) = batch.pad(train_events)
padded_X = np.expand_dims(padded_X,axis=2)

(padded_y,sizes) = batch.pad(train_bases)
padded_y = padded_y.astype(int)

# pass data to batch iterator class
dataset = batch.data_helper(padded_X, padded_y, small_batch=False, return_length=True)

# for outputing test accuracy
if args.test_data:
    (test_events, test_bases) = batch.load_data(args.test_data)

    # pad data and labels
    (padded_test_data,test_data_sizes) = batch.pad(test_events)
    padded_test_data = np.expand_dims(padded_test_data,axis=2)

    (padded_test_labels,test_data_sizes) = batch.pad(test_bases)
    padded_test_labels = padded_test_labels.astype(int)

# Set up the model
# data X are [BATCH_SIZE, MAX_INPUT_SIZE, 1]
# labels y are [BATCH_SIZE, MAX_INPUT_SIZE]
X = tf.placeholder(shape=[None, None, 1], dtype=tf.float32, name='X')
y = tf.placeholder(shape=[None, None], dtype=tf.int32, name='y')
sequence_length = tf.placeholder(shape=[None],dtype=tf.int32,name='sequence_length')

# use dynamic RNN to allow for flexibility in input size
cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS,state_is_tuple=False)
cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS,state_is_tuple=False)
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, dtype=tf.float32, time_major=False, sequence_length=sequence_length)
outputs = tf.concat(outputs, 2)

# dense layer connecting to output
logits = tf.contrib.layers.linear(outputs, NUM_OUTPUTS)
prediction = tf.add(tf.argmax(logits, 2),1, name='prediction') # adding one for 1-4096

# set up minimization of loss
# tf.one_hot(y-1...) converts 1-indexed labels to encodings (0 saved for padding)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
     labels=tf.one_hot(y-1, depth=NUM_OUTPUTS, dtype=tf.float32),logits=logits, name='cross_entropy')
loss = tf.reduce_mean(cross_entropy, name='loss')
train_op = tf.train.AdamOptimizer().minimize(loss)

# Start training network
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_counter = 0

    # main training loop
    for iteration in range(TRAINING_STEPS):

        # get current minibatch and run minimization
        (X_batch, y_batch, sequence_length_batch) = dataset.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict={X:X_batch, y:y_batch, sequence_length:sequence_length_batch})

        # periodically output the loss
        if (iteration+1) % LOSS_ITER == 0:
            print('iteration:',iteration+1,'epoch:',dataset.epoch,'loss:',sess.run(loss, feed_dict={X:X_batch, y:y_batch, sequence_length:sequence_length_batch}))

        # periodically output test and training loss
        if (iteration+1) % ACCURACY_ITER == 0:

            if args.test_data:
                predict_test = sess.run(prediction, feed_dict={X:padded_test_data, sequence_length:test_data_sizes})
                test_accuracy = accuracy(test_data_sizes, predict_test, padded_test_labels)

            predict_train = sess.run(prediction, feed_dict={X:padded_X, sequence_length:sizes})
            training_accuracy = accuracy(sizes, predict_train, padded_y)

            if args.test_data:
                print('iteration:',iteration+1,'epoch:',dataset.epoch,'training_accuracy:',training_accuracy, 'test_accuracy:',test_accuracy)
            else:
                print('iteration:',iteration+1,'epoch:',dataset.epoch,'training_accuracy:',training_accuracy)

          # periodically save the current model parameters
        if (iteration+1) % CHECKPOINT_ITER == 0:
            saver.save(sess, args.save_dir+'/model', global_step=checkpoint_counter)
            checkpoint_counter += 1

    # save extra model at the end of training
    saver.save(sess, args.save_dir+'/model', global_step=checkpoint_counter)
