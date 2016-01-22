#!/bin/bash

#run from ../
#decoding wrapper

nj=8
mono=false

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;
decoder=$1
srcdir=$2
datadir=$3


if [ $mono = true ];then
  echo  "using monophone to generate graph"
  opt="--mono"
fi
#decode word
utils/mkgraph.sh $opt data/graph/lang $srcdir $srcdir/graph.word  || exit 1;
$decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph.word $datadir/test $srcdir/decode_test_word || exit 1

#decode phone
utils/mkgraph.sh $opt data/graph.phone/lang $srcdir $srcdir/graph.phone  || exit 1;
$decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph.phone $datadir/test.ph $srcdir/decode_test_phone || exit 1


