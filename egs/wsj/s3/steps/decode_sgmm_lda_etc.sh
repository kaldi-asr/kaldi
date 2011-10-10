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

# Decoding script that works with an SGMM model... note: this script 
# assumes you have speaker vectors [for no vectors, see decode_sgmm_novec_lda_etc.sh,
# if it exists already].
# It works on top of LDA + [something] features; if this includes
# speaker-specific transforms, you have to provide an "old" decoding directory
# where the transforms are located.  The data decoded in that directory must be
# split up in the same way as the current directory.

if [ -f ./path.sh ]; then . ./path.sh; fi

nj=1
jobid=0
if [ "$1" == "-j" ]; then
  shift;
  nj=$1;
  jobid=$2;
  shift; shift;
  if [ $jobid -ge $nj ]; then
     echo "Invalid job number, $jobid >= $nj";
     exit 1;
  fi
fi

if [ $# -lt 3 -o $# -gt 4 ]; then
   echo "Usage: steps/decode_sgmm_lda_etc.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir> [<old-decode-dir>]"
   echo " e.g.: steps/decode_sgmm_lda_etc.sh -j 10 0 exp/sgmm3c/graph_tgpr data/dev_nov93 exp/sgmm3c/decode_dev93_tgpr exp/tri2b/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
olddir=$4
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
silphonelist=`cat $graphdir/silphones.csl`

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $nj -gt 1 ]; then
  mydata=$data/split$nj/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $srcdir/final.alimdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_sgmm_lda_etc.sh: no such file $f";
     exit 1;
  fi
done
if [ ! -z "$olddir" ]; then # "$olddir" nonempty..
  for n in `get_splits.pl $nj`; do
    if [ ! -f $olddir/$n.trans ]; then
      echo "Expect file $olddir/$n.trans to exist"
      exit 1
    fi
  done
fi

feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
[ ! -z "$olddir" ] && feats="$feats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$olddir/$jobid.trans ark:- ark:- |"


# Do Gaussian selection, since we'll have two decoding passes and don't want to redo this.
# Note: it doesn't make a difference if we use final.mdl or final.alimdl, they have the
# same UBM.
sgmm-gselect $srcdir/final.mdl "$feats" "ark:|gzip -c > $dir/$jobid.gselect.gz" \
    2>$dir/gselect$jobid.log || exit 1;
gselect_opt="--gselect=ark:gunzip -c $dir/$jobid.gselect.gz|"


# Generate a state-level lattice for rescoring, with the alignment model and no speaker
# vectors.

sgmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=$acwt  \
  --determinize-lattice=false --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  "$gselect_opt" $srcdir/final.alimdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/pre_lat.$jobid.gz" \
   2> $dir/decode_pass1.$jobid.log || exit 1;

( lattice-determinize --acoustic-scale=$acwt --prune=true --beam=4.0 \
     "ark:gunzip -c $dir/pre_lat.$jobid.gz|" ark:- | \
   lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   sgmm-post-to-gpost "$gselect_opt" $srcdir/final.alimdl "$feats" ark:- ark:- | \
   sgmm-est-spkvecs-gpost --spk2utt=ark:$mydata/spk2utt \
    $srcdir/final.mdl "$feats" ark:- "ark:$dir/$jobid.vecs" ) \
      2> $dir/vecs.$jobid.log || exit 1;

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.

sgmm-rescore-lattice "$gselect_opt" --utt2spk=ark:$mydata/utt2spk --spk-vecs=ark:$dir/$jobid.vecs \
  $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.$jobid.gz|" "$feats" \
 "ark:|lattice-determinize --acoustic-scale=$acwt --prune=true --beam=6.0 ark:- ark:- | gzip -c > $dir/lat.$jobid.gz" \
  2>$dir/rescore.$jobid.log || exit 1;

rm $dir/pre_lat.$jobid.gz

# The top-level decoding script will rescore "lat.$jobid.gz" to get the final output.
