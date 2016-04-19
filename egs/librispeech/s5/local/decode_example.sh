#!/bin/bash

# Copyright 2015  Guoguo Chen
# Apache 2.0
#
# Script that shows how to modify the language model, and how to run the
# decoding from scratch.

warn="WARNING: you are supposed to unpack the tar ball under egs/librispeech/s5"

if [ ! -f cmd.sh ]; then
  echo "cmd.sh file is missing"
  echo $warn && exit 1
fi
if [ ! -f path.sh ]; then
  echo "path.sh file is missing"
  echo $warn && exit 1
fi

. ./cmd.sh || exit 1
. ./path.sh || exit 1

# Data directory, if you want to decode other audio files, change here,
datadir=test_clean_example

# Language model, if you want ot decode with another language model, change
# here.
lm=data/local/lm/lm_tgsmall.arpa.gz

modeldir=exp/nnet2_online/nnet_ms_a_online/
mfccdir=mfcc

# Compiles language model, if you want to modify the language model, change
# here.
lang=data/lang
lang_test=data/lang_test
mkdir -p $lang_test
cp -r $lang/* $lang_test
gunzip -c $lm | arpa2fst --disambig-symbol=#0 \
                 --read-symbol-table=$lang_test/words.txt - $lang_test/G.fst
utils/validate_lang.pl --skip-determinization-check $lang_test || exit 1;

# Compiles decoding graph.
graphdir=$modeldir/graph_test
utils/mkgraph.sh $lang_test $modeldir $graphdir || exit 1;

steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 1 \
  $graphdir data/$datadir $modeldir/decode_${datadir}_test || exit 1;
