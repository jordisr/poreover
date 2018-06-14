![Logo](logo.png)
# PoreOver: Nanopore basecalling in TensorFlow
### Introduction
PoreOver is a neural network basecaller for the Oxford Nanopore sequencing platform and is under active development. It is intended as a platform on which to explore new algorithms and architectures for basecalling. The current version uses a bidirectional RNN with LSTM cells and CTC loss to call bases from raw signal, and has been inspired by other community basecallers such as DeepNano (Boža et al. 2017) and Chiron (Teng et al. 2018).

#### Consensus basecalling
Consensus basecalling, intended for higher accuracy 1D^2 sequencing, is currently being added to PoreOver, though these algorithms are still being tested and not yet ready for production use.

### Requirements
* TensorFlow (GPU installation recommended)
* Python 3

### Usage examples
Full options for `train_model.py` and `run_model.py` available with the `--help` flag.

First, we need to extract the toy training data set in `data/`
```
cd data
tar -xzf train.tar.gz
```

We can now train the model for a few iterations.

`python train_model.py --data data/train --training_steps 100`

Once there is a model to load, we can make a basecall on a sample read (of course,
    after only a little training on a toy dataset we would not expect it to be very accurate). 

`python run_model.py --fast5 data/read.fast5 --model run-0 --fasta`
 
We can compare this to the output of a model that has seen more training on a larger dataset. In the `models/` directory there are models for pore versions R9 (trained on E.coli) and R9.5 (trained on human).

`python run_model.py --fast5 data/read.fast5 --model models/r9 --fasta`
