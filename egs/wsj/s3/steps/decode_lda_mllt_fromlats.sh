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


# This decoding script does not decode from scratch, but limits the
# decoding to a set of lattices from another directory-- these lattices
# do not have to be built with the same tree, but of course the integer
# word id's do have to match, i.e. you have to use the same vocabulary.
# (in the WSJ setup this is not a problem as we have the same vocab
# throughout).
#
# If the tree was the same, you wouldn't need to use this mechanism,
# you could use gmm-rescore-lattice.
#
# The way this script works is to create a separate decoding graph for
# each file, limited to just the word-sequences present in the original
# lattice.  This is done by making an unweighted word acceptor from
# the lattice, composing with the grammar (G), determinizing, and using
# this as the utterance-specific grammar which is given to 
# compile-train-graphs-fsts.  Note: in this case we put the transition
# weights into the FSTs during compilation, which is different from during
# GMM alignment in training, where they are added by the decoding program
#
# This script assumes you have a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA+MLLT or similar transform.
# Its output is lattices, for a single broken-up piece of the data.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"

for n in 1 2; do
  if [ "$1" == "-j" ]; then
    shift;
    numjobs=$1; 
    jobid=$2;
    shift; shift;
  fi
  if [ "$1" == "--scale-opts" ]; then
     scale_opts="$2";
     shift; shift;
  fi
done

if [ $# != 4 ]; then
   echo "Usage: steps/decode_lda_mllt_fromlats.sh [-j num-jobs job-number] <lang> <data-dir> <decode-dir> <old-decode-dir>"
   echo " e.g.: steps/decode_lda_mllt_fromlats.sh -j 10 0 data/lang_test_tg data/test_dev93 exp/tri2b/decode_tgpr_dev93_fromlats exp/tri2a/decode_tgpr_dev93"
   exit 1;
fi


lang=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
olddir=$4

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $lang/G.fst $lang/L_disambig.fst $lang/phones_disambig.txt $olddir/lat.$jobid.gz"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# Note: we limit the batch-size to 75 to prevent memory blowup.
# In the normal decoding scripts there are only about 50 utterances
# per batch anyway.

( lattice-to-fst "ark:gunzip -c $olddir/lat.$jobid.gz|" ark:- | \
  fsttablecompose "fstproject --project_output=true $lang/G.fst | fstarcsort |" ark:- ark:- | \
  fstdeterminizestar ark:- ark:- | \
  compile-train-graphs-fsts --read-disambig-syms="grep \# $lang/phones_disambig.txt | awk '{print \$2}'|" \
    --batch-size=75 $scale_opts $srcdir/tree $srcdir/final.mdl $lang/L_disambig.fst ark:- ark:- |  \
  gmm-latgen-faster --max-active=7000 --beam=20.0 --lattice-beam=7.0 --acoustic-scale=0.083333 \
    --allow-partial=true --word-symbol-table=$lang/words.txt \
    $srcdir/final.mdl ark:- "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" ) \
     2> $dir/decode$jobid.log || exit 1;

