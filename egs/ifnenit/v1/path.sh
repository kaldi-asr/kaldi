#!/bin/bash

# path to Kaldi's root directory
# root=../../tools/kaldi-trunk/
# root=/export/b01/babak/kaldi/kaldi/
export KALDI_ROOT=`pwd`/../../..

export LD_LIBRARY_PATH=/home/dpovey/libs:$KALDI_ROOT/src/chainbin:/usr/local/lib:$LD_LIBRARY_PATH

export PATH=$KALDI_ROOT/src/bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/chain:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:utils:$KALDI_ROOT/src/fgmmbin:$KALDI_ROOT/src/sgmm2bin:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/sgmmbin:$KALDI_ROOT/src/lm:$KALDI_ROOT/src/latbin:$PATH 


export LC_ALL=C
export LC_LOCALE_ALL=C


