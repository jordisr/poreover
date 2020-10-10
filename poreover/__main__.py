'''
PoreOver
'''
import argparse, sys, glob, os, logging, progressbar

from poreover.network.network import call, train
from poreover.decoding.decode import decode
from poreover.decoding.pair_decode import pair_decode

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PoreOver: Consensus Basecalling for Nanopore Sequencing')
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required=True

    # Train
    parser_train = subparsers.add_parser('train', help='Train a neural network base calling model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.set_defaults(func=train)
    parser_train.add_argument('--data', help='Location of training data in compressed npz format', required=True)
    parser_train.add_argument('--name', default='run', help='Name of run')
    parser_train.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on')
    parser_train.add_argument('--save_every', type=int, default=1000, help='Frequency with which to save checkpoint files')
    parser_train.add_argument('--holdout', default=0.05, type=float, help='Fraction of training data to hold out for calculating test error')
    parser_train.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output minibatch loss')
    parser_train.add_argument('--ctc_merge_repeated', action='store_true', default=False, help='boolean option for tf.compat.v1.nn.ctc_loss')
    parser_train.add_argument('--model', default='conv1_bigru3', choices=['bigru3', 'conv1_bigru3', 'conv2_bigru3', 'conv1_gru5'], help='Neural network architecture')
    parser_train.add_argument('--restart', default=False, help='Trained model to load (if directory, loads latest from checkpoint file)')
    parser_train.add_argument('--batch_size', default=64, type=int, help='Minibatch size for training')
    parser_train.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser_train.add_argument('--seed', type=int, default=None, help='Explicitly set random seed')
    parser_train.add_argument('--num_neurons', type=int, default=128, help='Number of neurons in RNN layers')
    parser_train.add_argument('--kernel_size', type=int, default=9, help='Kernel size in Conv1D layer')
    parser_train.add_argument('--filters', type=int, default=256, help='Number of filters in Conv1D layer')

    # Call
    parser_call = subparsers.add_parser('call', help='Run basecalling forward pass on set of FAST5 reads', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_call.set_defaults(func=call)
    parser_call.add_argument('in', help='Single FAST5 file or directory of FAST5 files')
    parser_call.add_argument('--weights', default=None, help='Trained weights to load into model (if directory, loads latest from checkpoint file)')
    parser_call.add_argument('--model', help='Model config JSON file', default=None)
    parser_call.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale'], help='Type of preprocessing (should be same as training)')
    parser_call.add_argument('--use_id', default=False, action='store_true', help='Save logits by read ID instead of FAST5 filename')
    parser_call.add_argument('--dir', default='.', help='Directory to write logits to')
    parser_call.add_argument('--window', type=int, default=1000, help='Call read using chunks of this size')
    parser_call.add_argument('--format', choices=['csv', 'npy'], default='npy', help='Save softmax probabilities to CSV file or logits to binarized NumPy format')
    parser_call.add_argument('--no_stack', default=False, action='store_true', help='Basecall [1xSIGNAL_LENGTH] tensor instead of splitting it into windows (slower)')

    # Decode
    parser_decode = subparsers.add_parser('decode', help='Decode basecaller probabilities to a FASTA file')
    parser_decode.set_defaults(func=decode)
    parser_decode.add_argument('in', nargs='+', help='Probabilities to decode (either .npy from PoreOver/Bonito or HDF5/FAST5 from Flappie or Guppy)')
    parser_decode.add_argument('--out', default='out',help='Prefix for FASTA sequence output')
    parser_decode.add_argument('--basecaller', choices=['poreover', 'flappie', 'guppy', 'bonito'], help='Basecaller used to generate probabilitiess')
    parser_decode.add_argument('--algorithm', default='viterbi', choices=['viterbi' ,'beam', 'prefix'], help='')
    parser_decode.add_argument('--window', type=int, default=400, help='Use chunks of this size for prefix search')
    parser_decode.add_argument('--beam_width', type=int, default=25, help='Width for beam search')
    parser_decode.add_argument('--threads', type=int, default=1, help='Processes to use')

    # Pair decode
    parser_pair= subparsers.add_parser('pair-decode', help='1D2 consensus decoding of two output probabilities', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_pair.set_defaults(func=pair_decode)
    # general options
    parser_pair.add_argument('in', nargs='+', help='Softmax probabilities to decode (either .npy from PoreOver, or HDF5/FAST5 from Flappie or Guppy) or list of read pairs')
    parser_pair.add_argument('--dir', default='.', help='Base directory to look in for basecaller probabilities')
    parser_pair.add_argument('--basecaller', choices=['poreover', 'flappie', 'guppy', 'bonito'], help='Basecaller used to generate probabilitiess')
    parser_pair.add_argument('--reverse_complement', default=False, action='store_true', help='Whether to reverse complement the second sequence')
    parser_pair.add_argument('--out', default='out',help='Prefix for FASTA sequence output')
    parser_pair.add_argument('--threads', type=int, default=1, help='Processes to use')
    parser_pair.add_argument('--method', choices=['align', 'split', 'envelope'], default='envelope', help=argparse.SUPPRESS) # Method for dividing up search space (DEPRECATED)
    parser_pair.add_argument('--single', choices=['beam', 'viterbi'], default='viterbi', help='Algorithm for 1D basecalling (used to build alignment envelope)')
    parser_pair.add_argument('--logging', default="info", choices=['info', 'debug'], help='Level for logging')
    parser_pair.add_argument('--debug', default=False, action='store_true', help='Save intermediate objects to pickled file for debugging')
    parser_pair.add_argument('--algorithm', default='beam', choices=['prefix' ,'beam'], help=argparse.SUPPRESS) # Search algorithm for pair decoding
    parser_pair.add_argument('--alignment', default='banded', choices=['banded' ,'full'], help='Do full Needleman-Wunsch alignment between 1D basecalls to build envelope')
    parser_pair.add_argument('--beam_width', type=int, default=5, help='Width for beam search')
    parser_pair.add_argument('--debug_envelope', action='store_true', help=argparse.SUPPRESS) # just print out statistics on the alignment envelope and don't basecall
    # --method envelope
    parser_pair.add_argument('--diagonal_envelope', action='store_true', help='Use a simple diagonal band for the signal alignment envelope')
    parser_pair.add_argument('--diagonal_width', type=int, default=50, help='Width of diagonal band envelope')
    parser_pair.add_argument('--padding', type=int, default=5, help='Padding for building alignment envelope')
    parser_pair.add_argument('--skip_matches', action='store_true', help='Skip regions of sequence alignment with match columns greater than --skip_threshold')
    parser_pair.add_argument('--skip_threshold', type=int, default=10, help='Number of consecutive matches to use for --skip_matches')
    parser_pair.add_argument('--beam_search_method', choices=['row', 'row_col', 'grid'], default="row", help=argparse.SUPPRESS) # method for matrix traversal, passed to C++ decoder

    # --method split
    parser_pair.add_argument('--window', type=int, default=200, help=argparse.SUPPRESS) # Segment size used for splitting reads (DEPRECATED)

    # Parse arguments and call corresponding command
    args = parser.parse_args()
    args.func(args)

    print(args, file=sys.stderr)

if __name__ == "__main__":
    main()
