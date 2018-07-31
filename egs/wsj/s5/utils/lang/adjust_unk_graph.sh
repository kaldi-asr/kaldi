#!/bin/bash
# Copyright 2018 Xiaohui Zhang
# Apache 2.0

# This script copies a fully expanded decoding graph (HCLG.fst) and scales the scores
# of all arcs whose output symbol is a user-specified OOV symbol (or any other word).
# This achieves an equivalent effect of utils/lang/adjust_unk_arpa.pl, which scales
# the LM prob of all ngrams predicting an OOV symbol, while avoiding re-creating the graph.

set -o pipefail

if [ $# != 4 ]; then
   echo "Usage: utils/adjust_unk_graph.sh <oov-dict-entry> <scale> <in-graph-dir> <out-graph-dir>"
   echo "e.g.: utils/adjust_unk_graph.sh \"<unk>\" 0.1 exp/tri1/graph exp/tri1/graph_unk_scale_0.1"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

oov_word=$1
unk_scale=$2
graphdir_in=$3
graphdir_out=$4

mkdir -p $graphdir_out

required="HCLG.fst words.txt disambig_tid.int num_pdfs phones phones.txt words.txt"
for f in $required; do
  [ ! -f $graphdir_in/$f ] && echo "adjust_unk_graph.sh: expected $graphdir_in/$f to exist" && exit 1;
  cp -r $graphdir_in/$f $graphdir_out
done

cp -r $graphdir_in/{disambig_tid.int,num_pdfs,phones,phones.txt,words.txt} $graphdir_out

oov_id=`echo $oov_word | utils/sym2int.pl $graphdir_in/words.txt`
[ -z $oov_id ] && echo "adjust_unk_graph.sh: the specified oov symbol $oov_word is out of the vocabulary." && exit 1;
fstprint $graphdir_in/HCLG.fst | awk -v oov=$oov_id -v unk_scale=$unk_scale '{if($4==oov) $5=$5-log(unk_scale);print $0}' | \
  fstcompile > $graphdir_out/HCLG.fst || exit 1;
