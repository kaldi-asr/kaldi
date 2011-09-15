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

if [ $# != 4 -a $# != 5 ]; then
  echo "Usage: steps/decode_sgmm_lda_etc.sh <model-dir> <data-dir> <lang-dir> <decode-dir> [<old-decode-dir>]"
  echo " e.g.: steps/decode_sgmm_lda_etc.sh exp/sgmm3d data/test_feb89 data/lang_test exp/sgmm3d/decode_feb89"
  echo " or: steps/decode_sgmm_lda_etc.sh exp/sgmm3e data/test_feb89 data/lang_test exp/sgmm3e/decode_feb89 exp/tri2c/decode_feb89"
  exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
olddir=$5 # old decoding dir where there are transforms.
graphdir=$srcdir/graph

silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

# -f means file exists; -o means or.
requirements="$srcdir/final.mdl $srcdir/final.alimdl $srcdir/final.mat"

for f in $requirements; do
  if [ ! -f $f ]; then
    echo "decode_lda_etc.sh: input file $f does not exist";
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

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

if [ ! -z $olddir ]; then # i.e. if $olddir not empty string...
  if [ ! -f $olddir/trans.ark ]; then
     echo decode_sgmm_lda_etc.sh: error: no such file $olddir/trans.ark 
     exit 1
  fi
  feats="$feats transform-feats --utt2spk=ark:$data/utt2spk ark:$olddir/trans.ark ark:- ark:- |"
fi

sgmm-gselect $srcdir/final.mdl "$feats" "ark:|gzip -c > $dir/gselect.gz" \
    2>$dir/gselect.log || exit 1;
gselect_opt="--gselect=ark:gunzip -c $dir/gselect.gz|"

# Using smaller beam for first decoding pass.
sgmm-decode-faster "$gselect_opt" --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=$lang/words.txt \
  $srcdir/final.alimdl $graphdir/HCLG.fst "$feats" ark,t:$dir/pass1.tra ark,t:$dir/pass1.ali \
     2> $dir/decode_pass1.log || exit 1;

( ali-to-post ark:$dir/pass1.ali ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   sgmm-post-to-gpost $srcdir/final.alimdl "$feats" ark:- ark:- | \
   sgmm-est-spkvecs-gpost --spk2utt=ark:$data/spk2utt $srcdir/final.mdl "$feats" \
       ark,s,cs:- ark:$dir/pre_vecs.ark ) \
     2> $dir/vecs1.log || exit 1;

( ali-to-post ark:$dir/pass1.ali ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   sgmm-est-spkvecs --spk-vecs=ark:$dir/pre_vecs.ark --spk2utt=ark:$data/spk2utt \
      $srcdir/final.mdl "$feats" ark,s,cs:- ark:$dir/vecs.ark ) \
     2> $dir/vecs2.log || exit 1;


# Second pass decoding...
sgmm-latgen-simple --beam=20.0 --acoustic-scale=0.1 "$gselect_opt" \
  --spk-vecs=ark:$dir/vecs.ark --utt2spk=ark:$data/utt2spk \
  --word-symbol-table=$lang/words.txt $srcdir/final.mdl $graphdir/HCLG.fst \
  "$feats" "ark:|gzip -c >$dir/lat.gz" ark,t:$dir/pass2.tra ark,t:$dir/pass2.ali \
    2> $dir/decode_pass2.log || exit 1;



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

