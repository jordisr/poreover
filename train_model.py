'''
TODO
- double check loss expression
- load model to resume training
- add additional hidden layers
'''

import numpy as np
import tensorflow as tf
import argparse

# some custom helper functions
import batch
import kmer

def build_rnn(X,y):
    # model parameters
    NUM_NEURONS = 100 # how many neurons
    NUM_OUTPUTS = 4097 #number of possible 6-mers + NNNNNN

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

    return(train_op, loss)

# parse command line arguments
parser = argparse.ArgumentParser(description='Train the basecaller')
parser.add_argument('--data', help='Location of training data', required=True)
parser.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
parser.add_argument('--name', default='run', help='Name of run')
#parser.add_argument('--resume_from', help='Resume training from checkpoint file')

parser.add_argument('--training_steps', type=int, default=1000, help='Number of iterations to run training (default: 1000)')
parser.add_argument('--save_every', type=int, default=10000, help='Frequency with which to save checkpoint files (default: 10000)')
parser.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output minibatch loss')
#parser.add_argument('--loss_file',type=int,help='Write loss to a file instead of stdout')
args = parser.parse_args()

# user options
TRAINING_STEPS = args.training_steps #number of iterations of SGD
CHECKPOINT_ITER = args.save_every
LOSS_ITER = args.loss_every # how often to output loss
BATCH_SIZE = 32 # number of read fragments to use at a time

# load training data into memory (small files so this is OK for now)
INPUT_DIM = 2 # currently [event_level_mean, event_stdv]
(train_events, train_bases) = batch.load_data(args.data, INPUT_DIM)
EPOCH_SIZE = len(train_events)

# log file for training statuts
log_file = open(args.save_dir+'/'+args.name+'.out','w')

# pass data to batch iterator class
dataset = batch.data_helper(train_events, train_bases, small_batch=False, return_length=True)

# Set up the model
# data X are [BATCH_SIZE, MAX_INPUT_SIZE, 1]
# labels y are [BATCH_SIZE, MAX_INPUT_SIZE]
X = tf.placeholder(shape=[None, None, INPUT_DIM], dtype=tf.float32, name='X')
y = tf.placeholder(shape=[None, None], dtype=tf.int32, name='y')
sequence_length = tf.placeholder(shape=[None],dtype=tf.int32,name='sequence_length')
(train_op, loss) = build_rnn(X,y)

# Start training network
saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_counter = 0

    # main training loop
    for iteration in range(TRAINING_STEPS):

        # get current minibatch and run minimization
        (X_batch, y_batch, sequence_length_batch) = dataset.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict={X:X_batch, y:y_batch, sequence_length:sequence_length_batch})

        # periodically output the minibatch loss
        if (iteration+1) % LOSS_ITER == 0:
            print(iteration)
            log_file.write(batch.format_string(('iteration:',iteration+1,'epoch:',dataset.epoch,'minibatch_loss:',sess.run(loss, feed_dict={X:X_batch, y:y_batch, sequence_length:sequence_length_batch}))))

        # periodically save the current model parameters
        if (iteration+1) % CHECKPOINT_ITER == 0:
            saver.save(sess, args.save_dir+'/'+args.name, global_step=checkpoint_counter)
            log_file.write(batch.format_string(('iteration:',iteration+1,'epoch:',dataset.epoch, 'model:',checkpoint_counter)))
            checkpoint_counter += 1

    # save extra model at the end of training
    checkpoint_counter += 1
    saver.save(sess, args.save_dir+'/'+args.name, global_step=checkpoint_counter)
