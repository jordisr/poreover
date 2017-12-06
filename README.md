![Logo](logo.png)
# PoreOver: Nanopore basecalling in TensorFlow
### Introduction
PoreOver is an RNN basecaller for the Oxford Nanopore sequencing platform and is under active development. The current version uses a bidirectional RNN with LSTM cells and CTC loss to call bases from raw signal.

### Requirements
* TensorFlow 1.2 (GPU installation recommended)
* Python 3

### Training example
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
 
We can compare this to the output of a model that has seen more training on a larger dataset.

`python run_model.py --fast5 data/read.fast5 --model models/r9 --fasta`
