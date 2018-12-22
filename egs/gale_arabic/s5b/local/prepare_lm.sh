#!/bin/bash

# Copyright 2012  Vassil Panayotov
#           2017  Ewald Enzinger
# Apache 2.0

. ./path.sh || exit 1

echo "=== Building a language model ..."

locdata=data/local/lm/
mkdir -p $locdata

# Language model order
order=3

. utils/parse_options.sh

# Prepare a LM training corpus from the transcripts
mkdir -p $locdata

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

cat data/train/text | cut -d " " -f 2- >  $locdata/train.txt

ngram-count -text $locdata/train.txt -order $order -interpolate \
  -kndiscount -lm $locdata/lm.gz

#ngram -lm $locdata/lm.gz -ppl $locdata/dev.txt
echo "*** Finished building the LM model!"
