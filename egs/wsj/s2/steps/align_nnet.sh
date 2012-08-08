#!/bin/bash
# Copyright 2012 Karel Vesely

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

# To be run from ..

# This script does training-data alignment given a neural network
# built using CMN and TRAPs-DCT feature extraction.
# experimental directory, is ali, tree, final.nnet and final.mdl
# (the last three are just copied from the source directory). 

# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).


graphs=
while [ 1 ]; do
  case $1 in
    --graphs)
      shift; graphs=$1; shift;
      ;;
    --norm-vars)
      shift; norm_vars=$1; shift;
      ;;
    --splice-lr)
      shift; splice_lr=$1; shift;
      ;;
    *)
      break;
      ;;
  esac
done



if [ $# != 4 ]; then
   echo "Usage: steps/align_nnet.sh <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo " e.g.: steps/align_nnet.sh data/train data/lang exp/mono1a_nnet exp/mono1a_nnet_ali"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov_sym=`cat $lang/oov.txt`


######## CONFIGURATION
echo norm_vars ${norm_vars:=false} #false:CMN, true:CMVN on fbanks
echo splice_lr: ${splice_lr:=15}   #left- and right-splice value
transf=$srcdir/hamm_dct.mat #hamming DCT transform
cmvn_g="$srcdir/cmvn_glob.mat"
priors=$srcdir/cur.counts
nnet=$srcdir/final.nnet
########



mkdir -p $dir
cp $nnet $dir/final.nnet || exit 1;  # Create copy of that model...
nnet=$dir/final.nnet
cp $srcdir/tree $dir/tree || exit 1; # and the tree...
cp $srcdir/final.mdl $dir/final.mdl || exit 1; # and the transition model...

#TODO: same as GMMs?
scale_opts="--transition-scale=1.0 --acoustic-scale=0.12 --self-loop-scale=0.1"


# prepare features
# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp.fbank ark:- | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk ark:- scp:$data/feats.scp.fbank ark:- |"
# Splice+Hamming+DCT
feats="$feats splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- | transform-feats --print-args=false $transf ark:- ark:- |"
# Norm+MLP
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- | nnet-forward --silent=true --print-args=false --apply-log=true --class-frame-counts=$priors $nnet ark:- ark:- |" #use priors! no prior-scale...



# Align all training data using the supplied model.

echo "Aligning all training data"
if [ -z "$graphs" ]; then # --graphs option not supplied [-z means empty string]
  # compute integer form of transcripts.
  scripts/sym2int.pl --map-oov "$oov_sym" --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
    || exit 1;
  align-mapped $scale_opts --beam=8 --retry-beam=40 $dir/tree $dir/final.mdl $lang/L.fst \
   "$feats" ark:$dir/train.tra ark:$dir/ali 2> $dir/align.log || exit 1;
  rm $dir/train.tra
else
  align-compiled-mapped $scale_opts --beam=8 --retry-beam=40 $dir/final.mdl \
   "$graphs" "$feats" ark:$dir/ali 2> $dir/align.log || exit 1;
fi

if [ -z $dir/ali ]; then echo "Error, the alignments $dir/ali were not created..."; exit 1; fi
gzip -c $dir/ali > $dir/ali.gz

echo "Done."
