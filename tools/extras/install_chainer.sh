#!/bin/bash

# Installs chainer with nn-gev dependencies
# miniconda should be installed in $HOME/miniconda3/ 
# Download cuDNN from "https://developer.nvidia.com/rdp/cudnn-download" and extract in "$HOME/cuda/"
# and add their paths "export CFLAGS=-I$HOME/cuda/include" "export LDFLAGS=-L$HOME/cuda/lib64"

$HOME/miniconda3/bin/python -m pip install --user chainer==1.16.0
$HOME/miniconda3/bin/python -m pip install --user tqdm
$HOME/miniconda3/bin/python -m pip install --user scikit-learn
$HOME/miniconda3/bin/python -m pip install --user librosa
