'''
TODO
- double check loss expression
- load model to resume training
- add additional hidden layers
- output accuracy on training set periodically
- move model initialization and architecture to separate class for modularity
'''

import numpy as np
import tensorflow as tf
import argparse

# some custom helper functions
import batch
import kmer

# parse command line arguments
parser = argparse.ArgumentParser(description='Train the basecaller')
parser.add_argument('--data', help='Location of training data', required=True)
parser.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
#parser.add_argument('--resume_from', help='Resume training from checkpoint file')

parser.add_argument('--training_steps', type=int, default=1000, help='Number of iterations to run training (default: 1000)')
parser.add_argument('--save_every', type=int, default=10000, help='Frequency with which to save checkpoint files (default: 10000)')
parser.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output batch loss (default: 100)')
#parser.add_argument('--loss_file',type=int,help='Write loss to a file instead of stdout')
args = parser.parse_args()

# model parameters
BATCH_SIZE = 32 # number of read fragments to use at a time
NUM_NEURONS = 100 # how many neurons
NUM_LAYERS = 1 # NOT CURRENTLY USED
NUM_OUTPUTS = 4096 #number of possible 6-mers

# user options
TRAINING_STEPS = args.training_steps #number of iterations of SGD
CHECKPOINT_ITER = args.save_every #how often to output checkpoint files
LOSS_ITER = args.loss_every # how often to output loss

# training data
EVENTS_FILE = args.data+'.events' # event mean file
BASES_FILE = args.data+'.bases' # kmer file

# load training data into memory (small files so this is OK for now)
raw_events = []
raw_bases = []
with open(EVENTS_FILE,'r') as ef, open(BASES_FILE,'r') as bf:
    for eline, bline in zip(ef,bf):
        events = eline.split()
        bases = bline.split()
        if (len(events) == len(bases)):
            raw_events.append(np.array(list(map(lambda x: float(x),events))))
            raw_bases.append(np.array(list(map(kmer.kmer2label,bases))))

# pad data and labels
(padded_X,sizes) = batch.pad(raw_events)
padded_X = np.expand_dims(padded_X,axis=2)
(padded_y,sizes) = batch.pad(raw_bases)
#print("CHECKING SHAPES:",padded_X.shape, padded_y.shape)

# pass data to batch iterator class
dataset = batch.data_helper(padded_X, padded_y, small_batch=False, return_length=True)

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
saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

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

          # periodically save the current model parameters
        if (iteration+1) % CHECKPOINT_ITER == 0:
            saver.save(sess, args.save_dir+'/model', global_step=checkpoint_counter)
            checkpoint_counter += 1

    # save extra model at the end of training
    saver.save(sess, args.save_dir+'/model', global_step=checkpoint_counter)
