#!/bin/bash

DATA_DIR=$PWD/data
LEXICON=$DATA_DIR/local/dict/lexicon.txt
EXP_DIR="test"
TRAIN_DIR="train"

export KALDI_ROOT=/home/tools/kaldi
. $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$PWD/steps/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/nnet3bin/:$PWD:$PATH

export LC_ALL=C
#cd $WORK_DIR
