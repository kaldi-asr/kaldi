#!/bin/bash

# path to Kaldi's root directory
# root=../../tools/kaldi-trunk/
# root=/export/b01/babak/kaldi/kaldi/
export KALDI_ROOT=`pwd`/../../..

export LD_LIBRARY_PATH=/home/dpovey/libs:$KALDI_ROOT/src/chainbin:$LD_LIBRARY_PATH

export PATH=/home/babek/php/bin:/export/b01/babak/srilm/bin/:/export/b01/babak/srilm/bin/i686-m64/:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/chain:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:utils:$KALDI_ROOT/src/fgmmbin:$KALDI_ROOT/src/sgmm2bin:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/sgmmbin:$KALDI_ROOT/src/lm:$KALDI_ROOT/src/latbin:$PATH 
# path to the directory in which the subset of RM corpus is stored

export LC_ALL=en_US.utf8
export LC_LOCALE_ALL=C

export MONO_GAUSS="40000"
export TRI1_LEAVES="500"
export TRI1_GAUSS="55000"
export TRI2_LEAVES="750"
export TRI2_GAUSS="55000"
export TRI2_DIM="12"
export TRI3_LEAVES="750"
export TRI3_GAUSS="55000"
export nj=8

