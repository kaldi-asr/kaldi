#!/bin/bash
# Copyright 2018 Xiaohui Zhang
# Apache 2.0

# This script copies a fully expanded decoding graph (HCLG) and scale the scores
# of all arcs whose output symbol is a user-specified OOV symbol (or any other word).
# This achieves an equivalent effect of utils/lang/adjust_unk_arpa.pl, which scales
# the LM prob of all ngrams predicting an OOV symbol, while avoiding re-composing the graph.

set -o pipefail

if [ $# != 4 ]; then
   echo "Usage: utils/adjust_unk_graph.sh <oov-dict-entry> <scale> <in-graph-dir> <out-graph-dir>"
   echo "e.g.: utils/adjust_unk_graph.sh \"<unk>\" 0.1 exp/tri1/graph exp/tri1/graph_unk_scale_0.1"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

oov_word=$1
unk_scale=$2
idir=$3
odir=$4

mkdir -p $odir
cp -r $idir/* $odir

required="$idir/HCLG.fst $idir/words.txt"
for f in $required; do
  [ ! -f $f ] && echo "adjust_unk_graph.sh: expected $f to exist" && exit 1;
done

rm $odir/HCLG.fst || exit 1;
oov_id=`echo $oov_word | utils/sym2int.pl $idir/words.txt`
[ -z $oov_id ] && echo "adjust_unk_graph.sh: the specified oov symbol $oov_word is out of the vocabulary." && exit 1;
fstprint $idir/HCLG.fst | awk -v oov=$oov_id -v unk_scale=$unk_scale '{if($4==oov) $5=$5-log(unk_scale);print $0}' | \
  fstcompile > $odir/HCLG.fst || exit 1;
