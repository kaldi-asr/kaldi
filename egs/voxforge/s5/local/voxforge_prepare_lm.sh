#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

echo "=== Building a language model ..."

locdata=data/local
loctmp=$locdata/tmp

echo "--- Preparing a corpus from test and train transcripts ..."

# Language model order
order=3

. utils/parse_options.sh

# Prepare a LM training corpus from the transcripts _not_ in the test set
cut -f2- -d' ' < $locdata/test_trans.txt |\
  sed -e 's:[ ]\+: :g' | sort -u > $loctmp/test_utt.txt

# We are not removing the test utterances in the current version of the recipe
# because this messes up with some of the later stages - e.g. too many OOV
# words in tri2b_mmi
cut -f2- -d' ' < $locdata/train_trans.txt |\
   sed -e 's:[ ]\+: :g' |\
   sort -u > $loctmp/corpus.txt

if [ ! -f "tools/mitlm-svn/bin/estimate-ngram" ]; then
  echo "--- Downloading and compiling MITLM toolkit ..."
  mkdir -p tools
  command -v svn >/dev/null 2>&1 ||\
    { echo "SVN client is needed but not found" ; exit 1; }
  svn checkout http://mitlm.googlecode.com/svn/trunk/ tools/mitlm-svn
  cd tools/mitlm-svn/
  F77=gfortran ./autogen.sh
  ./configure --prefix=`pwd`
  make
  make install
  cd ../..
fi

echo "--- Estimating the LM ..."
if [ ! -f "tools/mitlm-svn/bin/estimate-ngram" ]; then
  echo "estimate-ngram not found! MITLM compilation failed?";
  exit 1;
fi
tools/mitlm-svn/bin/estimate-ngram -t $loctmp/corpus.txt -o $order \
 -write-vocab $locdata/vocab-full.txt -wl $locdata/lm.arpa

echo "*** Finished building the LM model!"
