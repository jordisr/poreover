# PoreOver: Nanopore basecalling in TensorFlow
### Introduction
PoreOver is an RNN basecaller for the Oxford Nanopore sequencing platform and is under active development. The current version uses a bidirectional RNN with LSTM cells to predict 6-mers for each event in a FAST5 file. 

### Requirements
* TensorFlow 1.2
* Python 3
* sense of adventure

### Usage
`python train_model.py --data_prefix data/toy --chk_dir models/`

`python run_model.py --model models/chkpt.sample --events data/read.fast5`

![Logo](logo.png)
