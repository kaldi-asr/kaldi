#!/usr/bin/env bash

# Copyright 2018 John Morgan
# Apache 2.0.

. ./cmd.sh
set -e
. ./path.sh
. $KALDI_ROOT/tools/env.sh
stage=0
nsegs=1000000;  # limit the number of training segments

. ./utils/parse_options.sh

if [ ! -d data/local/lm ]; then
    mkdir -p data/local/lm
fi

corpus=$1

if [ ! -f $corpus ]; then
  echo "$0: input data $corpus not found."
  exit 1
fi

perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < $corpus | \
     head -n $nsegs > data/local/lm/train.txt

if ! command ngram-count >/dev/null; then
  if uname -a | grep darwin >/dev/null; then # For MACOSX...
    sdir=$KALDI_ROOT/tools/srilm/bin/macosx
  elif uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
  else
    sdir=$KALDI_ROOT/tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi


ngram-count -order 3 -interpolate -unk -map-unk "<UNK>" \
    -limit-vocab -text data/local/lm/train.txt -lm data/local/lm/trigram.arpa || exit 1;

gzip -f data/local/lm/trigram.arpa
