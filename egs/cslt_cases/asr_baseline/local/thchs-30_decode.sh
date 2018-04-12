#!/bin/bash
#Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.

#decoding wrapper for thchs30 recipe
#run from ../

nj=8

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


decoder=$1
srcdir=$2
datadir=$3


###### Bookmark: graph making ######
utils/mkgraph.sh data/graph/lang $srcdir $srcdir/graph_word  || exit 1;


###### Bookmark: GMM decoding ######
$decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph_word $datadir/test $srcdir/decode_test_word || exit 1

