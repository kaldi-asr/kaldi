#!/bin/bash 

# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Decoding script that works with a SGMM model [w/ speaker vectors]
# and cepstral mean subtraction plus splice-9-frames plus LDA+MLLT, or
# LDA+MLLT+SAT or LDA+ET features.  For the last two, which
# are speaker adaptive, the script takes an extra argument 
# corresponding to the previous decoding directory where we can
# find the transform trans.ark.

# This script itself does two passes of decoding.

if [ $# != 5 -a $# != 6 ]; then
  echo "Usage: steps/decode_sgmm_lda_etc_fmllr.sh <model-dir> <data-dir> <lang-dir> <decode-dir> <old-sgmm-decode-dir> [<old-decode-dir-for-transforms>]"
  echo " e.g.: steps/decode_sgmm_lda_etc_fmllr.sh exp/sgmm3d data/test_feb89 data/lang_test exp/sgmm3d/decode/feb89 exp/sgmm3d/decode_fmllr/feb89"
  echo " or: steps/decode_sgmm_lda_etc_fmllr.sh exp/sgmm3e data/test_feb89 data/lang_test exp/sgmm3e/decode/feb89 exp/sgmm3e/decode_fmllr/feb89 exp/tri2c/decode/feb89"
  exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
firstpassdir=$5
olddir=$6 # old decoding dir where there are transforms [possibly]
graphdir=$srcdir/graph

silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

# -f means file exists; -o means or.
requirements="$srcdir/final.mdl $srcdir/final.fmllr_mdl $srcdir/final.mat $firstpassdir/cmvn.ark $firstpassdir/lat.gz $firstpassdir/gselect.gz $firstpassdir/vecs.ark"

for f in $requirements; do
  if [ ! -f $f ]; then
    echo "decode_lda_etc.sh: input file $f does not exist";
    exit 1;
  fi
done


feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$firstpassdir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

if [ ! -z $olddir ]; then # i.e. if $olddir not empty string...
  if [ ! -f $olddir/trans.ark ]; then
     echo decode_sgmm_lda_etc.sh: error: no such file $olddir/trans.ark 
     exit 1
  fi
  feats="$feats transform-feats --utt2spk=ark:$data/utt2spk ark:$olddir/trans.ark ark:- ark:- |"
fi

gselect_opt="--gselect=ark:gunzip -c $firstpassdir/gselect.gz|"


# Here we estimate the fMLLR transforms-- just one iteration should be sufficient,
# as it's after many adaptation passes.


( lattice-to-post --acoustic-scale=0.1 "ark:gunzip -c $firstpassdir/lat.gz|" ark:- | \
  weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- | \
  sgmm-est-fmllr --fmllr-iters=10 --fmllr-min-count=1000 "$gselect_opt" \
    --spk-vecs=ark:$firstpassdir/vecs.ark --spk2utt=ark:$data/spk2utt $srcdir/final.fmllr_mdl \
     "$feats" ark,s,cs:- ark:$dir/trans.ark ) 2>$dir/est_fmllr.log || exit 1;
  
feats="$feats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/trans.ark ark:- ark:- |"

sgmm-rescore-lattice "$gselect_opt" --spk-vecs=ark:$firstpassdir/vecs.ark \
  --utt2spk=ark:$data/utt2spk $srcdir/final.mdl \
  "ark:gunzip -c $firstpassdir/lat.gz|" "$feats" "ark:|gzip -c >$dir/lat.gz" \
  2>$dir/acoustic_rescore.log || exit 1;


# Now rescore lattices with various acoustic scales, and compute the WER.
for inv_acwt in 4 5 6 7 8 9 10; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$lang/words.txt \
     "ark:gunzip -c $dir/lat.gz|" ark,t:$dir/${inv_acwt}.tra \
     2>$dir/rescore_${inv_acwt}.log

  scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
   compute-wer --mode=present ark:-  ark,p:$dir/${inv_acwt}.tra \
    >& $dir/wer_${inv_acwt}
done

