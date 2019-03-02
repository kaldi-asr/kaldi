#!/bin/bash

# Copyright 2018  Govivace Inc. (Author: Valluri Saikiran)
# Apache License 2.0

# This script applies a g2p model using CMUsphinx/seq2seq.

stage=0
encoding='utf-8'

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <lexicon-in> <work-dir> <outdir>"
  echo "    where <lexicon-in> is the training lexicon (one pronunciation per "
  echo "    word per line, with lines like 'hello h uh l ow') and"
  echo "    <work-dir> is directory where the models will be stored"
  exit 1;
fi

lexicon=$1
wdir=$2
outdir=$3

mkdir -p $outdir

[ ! -f $lexicon ] && echo "Cannot find $lexicon" && exit

if [ ! -s `which g2p-seq2seq` ] ; then
  echo "g2p-seq2seq was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_g2p_seq2seq.sh"
  exit 1
fi

g2p-seq2seq --decode $lexicon --model_dir $wdir --output $outdir/lexicon.lex

