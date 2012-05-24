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

# To be run from .. (one directory up from here)

if [ "$1" == "--bn-dim" ]; then
  shift;
  bndim=$1
  shift;
fi


if [ $# != 5 ]; then
   echo "usage: make_bnfea.sh [--bn-dim N] <data-dir> <nnet-dir> <log-dir> <abs-path-to-bnfeadir> <num-cpus>";
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
nndir=$2
logdir=$3
bnfeadir=$4
ncpus=$5

######## CONFIGURATION
norm_vars=false #false:CMN, true:CMVN on fbanks
splice_lr=15   #left- and right-splice value
transf=$nndir/hamm_dct.mat
cmvn_g=$nndir/cmvn_glob.mat

#default options
echo bndim: ${bndim:=-1} #dimensionality of bottleneck, default disables trimming
########

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $bnfeadir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/feats.scp.fbank
required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_bnfea.sh: no such file $f"
    exit 1;
  fi
done

if [ ! -d $data/split$ncpus -o $data/split$ncpus -ot $data/feats.scp.fbank ]; then
  scripts/split_data.sh $data $ncpus
fi


#cut the MLP
nnet=$logdir/feature_extractor.nnet
nnet-copy --print-args=false --binary=false $nndir/final.nnet - 2>$logdir/nnet-copy.log | \
  awk '{ if(match($0,/<biasedlinearity> [0-9]+ '$bndim'/)) {stop=1;} if(stop==0) {print;}}' \
  > $nnet

rm $logdir/.error 2>/dev/null

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.
for ((n=0; n<ncpus; n++)); do
  log=$logdir/make_bnfea.$n.log

  # prepare features
  # We only do one forward pass, so there is no point caching the
  # CMVN stats-- we make them part of a pipe.
  feats="ark:compute-cmvn-stats --spk2utt=ark:$data/split$ncpus/$n/spk2utt scp:$data/split$ncpus/$n/feats.scp.fbank ark:- | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/split$ncpus/$n/utt2spk ark:- scp:$data/split$ncpus/$n/feats.scp.fbank ark:- |"
  # Splice+Hamming+DCT
  feats="$feats splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- | transform-feats --print-args=false $transf ark:- ark:- |"
  # Norm
  feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"

  # MLP forward 
  nnet-forward $nnet "$feats" \
   ark,scp:$bnfeadir/raw_bnfea_$name.$n.ark,$bnfeadir/raw_bnfea_$name.$n.scp \
   2> $log || touch $logdir/.error &
 
done
wait;

if [ -f $logdir/.error ]; then
  echo "Error producing bnfea features for $name:"
  tail $logdir/make_bnfea.*.log
  exit 1;
fi

# concatenate the .scp files together.
rm $logdir/feats.scp 2>/dev/null
for ((n=0; n<ncpus; n++)); do
  cat $bnfeadir/raw_bnfea_$name.$n.scp >> $logdir/feats.scp
done
#copy rest of the files to bnfeadir....
cp $data/{spk2gender,utt2spk,spk2utt,wav.scp,text} $logdir


echo "Succeeded creating MLP-BN features for $name"

