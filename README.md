![Logo](logo.png)
# PoreOver: Nanopore basecalling and consensus decoding
## Introduction

PoreOver is a basecalling tool for the Oxford Nanopore sequencing platform and is primarily intended for the task of consensus decoding raw basecaller probabilities for [higher accuracy 1D<sup>2</sup> sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1).
PoreOver includes a standalone RNN basecaller (PoreOverNet) that can be used to generate these probabilities, though the highest consensus accuracy is achieved in combination with [Bonito](https://github.com/nanoporetech/bonito), one of ONT's research basecallers.

More generally, PoreOver can serve as platform on which to explore new decoding algorithms and basecalling architectures.

If you find it useful, please cite:
~~~
Pair consensus decoding improves accuracy of neural network basecallers for nanopore sequencing.
Jordi Silvestre-Ryan, Ian Holmes. bioRxiv 2020.02.25.956771; doi: https://doi.org/10.1101/2020.02.25.956771
~~~

### Requirements
* Python 3
* TensorFlow 2

### Installation

~~~
git clone https://github.com/jordisr/poreover
cd poreover
pip install -r requirements.txt
pip install -e .
~~~

Now the software can be run with:

`poreover --help`

## Usage examples

PoreOver has four main modes:

* `call` Run a forward pass of PoreOver's neural network and save the probabilities
* `decode` Decode the output of PoreOver or another CTC basecaller using Viterbi or beam search
* `pair-decode` Generate consensus sequences given a list of paired read probabilities
* `train`: Train a new neural network using PoreOver

### Basecalling with PoreOver's network (PoreOverNet)

PoreOver includes a simple basecalling network with an architecture inspired by other community basecallers such as [DeepNano](https://bitbucket.org/vboza/deepnano/src/master/) (Boža et al. 2017) and [Chiron](https://github.com/haotianteng/Chiron) (Teng et al. 2018). It uses a single convolutional layer followed by three bidirectional GRU layers, and is trained with CTC loss. It is not as accurate as Bonito and is intended mostly for testing.

`poreover call data/read.fast5`

This will run the forward pass of the network and save the logits output to `read.npy`. This can then be passed to the `decode`

It will also take a directory as an input, e.g.
`poreover call data/1d2`

### Decoding basecaller probabilities

A nucleotide sequence can then be decoded to FASTA format using `decode`. The basecaller must be specified to correctly parse the file.

`poreover decode read.npy --basecaller poreover`

By default, this does Viterbi (i.e. best path) decoding, though alternatively `--algorithm beam` will perform a beam search, with the beam width configurable with `--beam_width`.
While beam search may outperform Viterbi decoding, in our experience any improvement is not usually worth the increased computational cost.

### Pair decoding of 1D<sup>2</sup> reads

Pair decoding can run either on a single pair as in

`poreover pair-decode data/1d2/read1.npy data/1d2/read2.npy --reverse_complement --basecaller poreover `

Or using a list of read pairs

`poreover pair-decode data/pairs.txt --reverse_complement --basecaller poreover`


### Modifying Bonito for use with PoreOver

As Bonito does not currently support saving the basecaller probabilities, a slight modification must be made to allow this. This can be done using the bonito_022.patch file (needs Bonito version 0.2.2).

~~~
git clone https://github.com/nanoporetech/bonito --branch v0.2.2
cd bonito
git apply poreover/data/bonito_022.patch
pip install -r requirements.txt
pip install -e .
~~~

### Decoding flip-flop basecallers

Flip-flop is a CTC variation developed by ONT and implemented in their production basecaller Guppy, as well as the research bascaller Flappie [Flappie](https://github.com/nanoporetech/flappie). Both of these basecallers can optionally output a trace of probabilities using the `--fast5-out` (in Guppy) or the `--trace` (Flappie) option.

PoreOver can read and decode these probabilities, yielding a basecalled sequence.

`poreover decode data/flappie_trace.hdf5`

`poreover decode data/guppy_flipflop.fast5`

Note: Beam search decoding (and by extension pair decoding) does not seem to perform well on the flip-flop model and so is not recommended.

### Training a new basecalling model for PoreOver

It is possible to use one of the architectures available in PoreOver to train a new basecalling model.

`poreover train --data data/training.npz --loss_every 5 --epochs 10`

This will use the default architecture of 1 convolutional + 3 bidirectional GRU layers, though there are a few other named architectures listed in `poreover/network/network.py`.

Once there is a model to load, we can make a basecall on a sample read (of course, after only a little training on a toy dataset we would not expect it to be very accurate).

`poreover call --fast5 data/read.fast5 --weights $RUN_DIRECTORY --model $RUN_DIRECTORY/model.json`
