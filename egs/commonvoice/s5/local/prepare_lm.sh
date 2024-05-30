#!/usr/bin/env bash

# Copyright 2012  Vassil Panayotov
#           2017  Ewald Enzinger
# Apache 2.0

# Adapted from egs/voxforge/s5/prepare_lm.sh (commit 1d9b858adc3e23bdbd9bf30231923755a1813cd0)

. ./path.sh || exit 1

echo "=== Building a language model ..."

data=data/valid_train
locdata=data/local
# Language model order
order=3

. utils/parse_options.sh

# Prepare a LM training corpus from the transcripts
mkdir -p $locdata
cut -f2- -d' ' < $data/text |\
  sed -e 's:[ ]\+: :g' | sort -u > $locdata/corpus.txt

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
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

ngram-count -order $order -write-vocab $locdata/vocab-full.txt -wbdiscount \
  -text $locdata/corpus.txt -lm $locdata/lm.gz

echo "*** Finished building the LM model!"
