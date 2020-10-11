import numpy as np
import h5py
import sys
import os
import glob
from pathlib import Path
from scipy.special import logsumexp

from multiprocessing import Pool, get_logger
import logging
import copy
import progressbar
from itertools import starmap

from . import prefix_search
from . import decoding_cpp
from . import transducer
import poreover.network as network

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

def softmax(logits):
    dim = len(logits.shape)
    axis_to_sum = dim-1
    return( (np.exp(logits).T / np.sum(np.exp(logits),axis=axis_to_sum).T).T )

def logit_to_log_likelihood(logits):
    # Normalizes logits so they are valid log-likelihoods
    # takes the place of softmax operation in data preprocessing
    dim = len(logits.shape)
    axis_to_sum = dim-1
    return( (logits.T - logsumexp(logits,axis=2).T).T )

def load_logits(file_path, flatten=False):
    read_reshape = np.load(file_path)
    if np.isclose(np.sum(read_reshape[0]), 1):
        #print('WARNING: Logits appear to be probabilities. Taking log.',file=sys.stderr)
        read_reshape = np.log(read_reshape)
    else:
        read_reshape = logit_to_log_likelihood(read_reshape)
    if flatten and len(read_reshape.shape) > 2:
        return(np.concatenate(read_reshape))
    else:
        return(read_reshape)

def trace_from_flappie(p):
    hdf = h5py.File(p, 'r')
    read_id = list(hdf)[0]
    trace = np.array(hdf[read_id]['trace'])
    # signal = np.array(hdf[read_id]['signal'])
    hdf.close()
    return(trace)

def trace_from_guppy(p):
    hdf = h5py.File(p, 'r')
    trace = np.array(hdf['/Analyses/Basecall_1D_000/BaseCalled_template/Trace'])
    hdf.close()
    return(trace)

def model_from_trace(f, basecaller=""):
    # infer model type from file
    file_name, file_extension = os.path.splitext(f)
    if file_extension == '.npy' and basecaller == 'poreover':
        try:
            trace = load_logits(f, flatten=True)
            model = transducer.poreover(trace)
        except:
            raise
    elif file_extension == '.npy' and basecaller == 'bonito':
            try:
                trace = load_logits(f, flatten=True)
                trace = trace[::,[1,2,3,4,0]]
                model = transducer.bonito(trace)
            except:
                raise
    elif file_extension == '.csv':
        trace = np.log(np.loadtxt(f, delimiter=',', skiprows=1))
        if trace.shape[1] == 5:
            model = transducer.poreover(trace)
        elif trace.shape[1] == 8:
            model = transducer.flipflop(trace)
    elif file_extension == '.hdf5' or basecaller == 'flappie':
        try:
            trace = trace_from_flappie(f)
            eps = 0.0000001
            trace = np.log((trace + eps)/(255 + eps))
            model = transducer.flipflop(trace)
        except:
            raise
    elif file_extension == '.fast5' or basecaller == 'guppy':
        try:
            trace = trace_from_guppy(f)
            eps = 0.0000001
            trace = np.log((trace + eps)/(255 + eps))
            model = transducer.flipflop(trace)
        except:
            raise
    else:
        if basecaller == "":
            print("Problem loading the trace probabilities, please specify where they came from with --basecaller [poreover/guppy/flappie]")
        else:
            print("Problem loading the trace probabilities")
        sys.exit(1)

    return(model)

def decode(args):

    # set up logger - should make it global
    progressbar.streams.wrap_stderr()
    #logging.basicConfig()
    logger = get_logger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # print software message, should incorporate to other subroutines as well
    coffee_emoji = u'\U00002615'
    dna_emoji = u'\U0001F9EC'
    logger.info('{0:2}{1:3}{0:2} {2:^30} {0:2}{1:3}{0:2}'.format(coffee_emoji, dna_emoji,'PoreOver decode'))

    # collect files for decoding
    in_path = getattr(args, 'in')
    in_files = in_path

    if len(in_path) == 1:
        if os.path.isdir(in_path[0]):
            file_ext = {'guppy':'.fast5','flappie':'.hdf5','bonito':'.npy','poreover':'.npy'}[args.basecaller]
            in_files = glob.glob("{}/*{}".format(in_path[0], file_ext))

    if len(in_files) > 1:
        # set up progressbar and manage output
        class callback_helper:
            def __init__(self):
                self.counter = 0
                self.pbar = progressbar.ProgressBar(max_value=len(in_files))
                self.out_f = open(args.out+'.fasta','w')
            def callback(self, x):
                self.counter += 1
                self.pbar.update(self.counter)
                print(x, file=self.out_f)
        callback_helper_ = callback_helper()

        bullet_point = u'\u25B8'+" "
        logger.info(bullet_point + "found {} reads to decode".format(len(in_files)))
        logger.info(bullet_point + "writing sequences to {0}.fasta".format(args.out))
        logger.info(bullet_point + "starting {} decoding processes...".format(args.threads))

        with Pool(processes=args.threads) as pool:
            for p in in_files:
                pool.apply_async(decode_helper, (p, args,), callback=callback_helper_.callback)
            pool.close()
            pool.join()

    else:
        seqs = decode_helper(in_path[0], args)
        with open(args.out+'.fasta', 'w') as out_fasta:
            print(seqs, file=out_fasta)

def decode_helper(in_path, args):
    # load probabilities from running basecaller
    model = model_from_trace(in_path, args.basecaller)
    model_type = {'poreover':'ctc','bonito':'ctc_merge_repeats','guppy':'ctc_flipflop','flappie':'ctc_flipflop','flipflop':'ctc_flipflop'}

    # call appropriate decoding function
    if args.algorithm == 'viterbi':
        sequence = model.viterbi_decode()
    elif args.algorithm == 'beam':
        sequence = decoding_cpp.cpp_beam_search(model.log_prob, args.beam_width, "ACGT", model_type[model.kind])
    elif args.algorithm == 'prefix':
        assert(model.kind == "poreover")
        # for comparing with previous results
        window = args.window
        i = 0
        sequence = ""
        while i+window < model.t_max:
            sequence += prefix_search.prefix_search_log_cy(model.log_prob[i:i+window])[0]
            i += window
        sequence += prefix_search.prefix_search_log_cy(model.log_prob[i:])[0]

    # output decoded sequence
    fasta_header = os.path.basename(in_path)
    return fasta_format(Path(in_path).stem, sequence)
