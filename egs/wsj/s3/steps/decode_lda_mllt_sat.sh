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

# Decoding script that works with a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA + MLLT + fMLLR features.  This script first
# generates a pruned state-level lattice without adaptation,
# then does acoustic rescoring on this lattice to generate
# a new lattice; it determinizes and prunes this ready for
# further rescoring (e.g. with new LMs, or varying the acoustic
# scale).

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
if [ "$1" == "-j" ]; then
  shift;
  numjobs=$1;
  jobid=$2;
  shift; shift;
  if [ $jobid -ge $numjobs ]; then
     echo "Invalid job number, $jobid >= $numjobs";
     exit 1;
  fi
fi

if [ $# != 3 ]; then
   echo "Usage: steps/decode_lda_mllt_sat.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt_sat.sh -j 10 0 exp/tri2c/graph_tgpr data/dev_nov93 exp/tri2c/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
acwt=0.08333 # Just used for adaptation and beam-pruning..
silphonelist=`cat $graphdir/silphones.csl`

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $srcdir/final.alimdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# basefeats is the speaker independent features.
basefeats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# Generate a state-level lattice for rescoring, so we don't have to redo the search
# after ET.
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=$acwt  \
  --determinize-lattice=false --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.alimdl $graphdir/HCLG.fst "$basefeats" "ark:|gzip -c > $dir/pre_lat.$jobid.gz" \
   2> $dir/decode_pass1.$jobid.log || exit 1;

(  lattice-determinize --acoustic-scale=$acwt --prune=true --beam=4.0 \
     "ark:gunzip -c $dir/pre_lat.$jobid.gz|" ark:- | \
   lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   gmm-post-to-gpost $srcdir/final.alimdl "$basefeats" ark:- ark:- | \
   gmm-est-fmllr-gpost --spk2utt=ark:$mydata/spk2utt $srcdir/final.mdl "$basefeats" \
       ark,s,cs:- ark:$dir/$jobid.trans ) \
    2> $dir/fmllr.$jobid.log || exit 1;

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.

feats="$basefeats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$dir/$jobid.trans ark:- ark:- |"

gmm-rescore-lattice $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.$jobid.gz|" "$feats" \
 "ark:|lattice-determinize --acoustic-scale=$acwt --prune=true --beam=6.0 ark:- ark:- | gzip -c > $dir/lat.$jobid.gz" \
  2>$dir/rescore.$jobid.log || exit 1;

rm $dir/pre_lat.$jobid.gz

# The top-level decoding script will rescore "lat.$jobid.gz" to get the final output.
