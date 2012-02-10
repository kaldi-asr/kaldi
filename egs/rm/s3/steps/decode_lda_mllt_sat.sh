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

# Decoding script that works with a GMM model and cepstral
# mean subtraction plus splice-9-frames plus LDA + MLLT + SAT features.
# Two passes of decoding.
# Used, for example, to decode tri3d/.

if [ $# != 4 ]; then
   echo "Usage: steps/decode_lda_mllt_sat.sh <model-dir> <data-dir> <lang-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt_sat.sh exp/tri2c data/test_feb89 data/lang_test exp/tri2c/decode_feb89"
   exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
graphdir=$srcdir/graph

silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

# -f means file exists; -o means or.
requirements="$srcdir/final.mdl $srcdir/final.alimdl $srcdir/final.mat"

for f in $requirements; do
  if [ ! -f $f ]; then
    echo "decode_lda_mllt_sat.sh: input file $f does not exist";
    exit 1;
  fi
done

if [ ! -f $graphdir/HCLG.fst -o $graphdir/HCLG.fst -ot $srcdir/final.mdl ]; then
   echo "Graph $graphdir/HCLG.fst does not exist or is too old."
   exit 1;
fi

# Compute CMVN stats.
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,t:$dir/cmvn.ark \
   2>$dir/cmvn.log

sifeats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# For Resource Management, we use beam of 20 and acwt of 0.1 by default
# (we tune over the accoustic weight during lattice rescoring).
# More normal, LVCSR setups would have a beam of 13 and acwt of 1/15 or so.
# If you decode with a beam of 20 on an LVCSR setup it will be very slow.

# This first pass decoding is quite fast; it's just done to get an initial estimate of the
# of the fMLLR transform.

gmm-latgen-simple --beam=15.0 --acoustic-scale=0.1 --word-symbol-table=$lang/words.txt \
  $srcdir/final.alimdl $graphdir/HCLG.fst "$sifeats" "ark:|gzip -c >$dir/pre_lat1.gz" \
     2> $dir/decode_pass1.log || exit 1;

adaptmdl=$srcdir/final.mdl # Compute fMLLR transforms with this model.
[ -f $srcdir/final.adaptmdl ] && adaptmdl=$srcdir/final.adaptmdl # e.g. in MMI-trained systems

(  gunzip -c $dir/pre_lat1.gz | \
   lattice-to-post --acoustic-scale=0.1 ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   gmm-post-to-gpost $srcdir/final.alimdl "$sifeats" ark:- ark:- | \
   gmm-est-fmllr-gpost --spk2utt=ark:$data/spk2utt $adaptmdl "$sifeats" \
       ark,s,cs:- ark:$dir/pre_trans.ark ) \
    2> $dir/fmllr1.log || exit 1;

pre_feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/pre_trans.ark ark:- ark:- |"

# Second pass decoding and lattice generation...
# here we're generating a state-level lattice, which allows for more exact
# acoustic lattice rescoring.
# Do this with SAT features with an initial, rough estimation from the SI decoding.

gmm-latgen-simple --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=$lang/words.txt \
   --determinize-lattice=false --allow-partial=true \
  $adaptmdl $graphdir/HCLG.fst "$pre_feats" "ark:|gzip -c > $dir/pre_lat2.gz" \
  2> $dir/decode_pass2.log || exit 1;

# Estimate the fMLLR transform once more.

( lattice-determinize --acoustic-scale=0.1 --prune=true --beam=4.0 \
     "ark:gunzip -c $dir/pre_lat2.gz|" ark:- | \
   lattice-to-post --acoustic-scale=0.1 ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $adaptmdl ark:- ark:- | \
   gmm-est-fmllr --spk2utt=ark:$data/spk2utt $adaptmdl "$pre_feats" \
      ark,s,cs:- ark:$dir/trans.tmp.ark ) \
    2> $dir/fmllr2.log || exit 1;

rm $dir/pre_lat1.gz

compose-transforms --b-is-affine=true ark:$dir/trans.tmp.ark ark:$dir/pre_trans.ark \
    ark:$dir/trans.ark 2>$dir/compose_transforms.log || exit 1;
#rm $dir/pre_trans.ark $dir/trans.tmp.ark || exit 1;

feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/trans.ark ark:- ark:- |"

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.

gmm-rescore-lattice $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat2.gz|" "$feats" \
 "ark:|lattice-determinize --acoustic-scale=0.1 --prune=true --beam=10.0 ark:- ark:- | gzip -c > $dir/lat.gz" \
  2>$dir/rescore.log || exit 1;

rm $dir/pre_lat2.gz

# Now rescore lattices with various acoustic scales, and compute the WERs.
for inv_acwt in 4 5 6 7 8 9 10; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$lang/words.txt \
     "ark:gunzip -c $dir/lat.gz|" ark,t:$dir/${inv_acwt}.tra \
     2>$dir/rescore_${inv_acwt}.log

  scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
   compute-wer --mode=present ark:-  ark,p:$dir/${inv_acwt}.tra \
    >& $dir/wer_${inv_acwt}
done


