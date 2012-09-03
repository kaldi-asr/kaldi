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

cmd=scripts/run.pl
nj=4
trim_transforms=4

while [ 1 ]; do
  case $1 in
    --cmd)
      shift; cmd=$1; shift;
    ;;
    --num-jobs)
      shift; nj=$1; shift;
    ;;
    --trim-transforms)
      shift; trim_transforms=$1; shift;
    ;;
    --*)
      echo "Unknown option $1";
      exit 1;
    ;;
    *)
      break;
    ;;
  esac
done
      


if [ $# != 4 ]; then
   echo "usage: $0 [--num-jobs N ] [--cmd CMD] [--trim-transforms N] <tgt-data-dir> <src-data-dir> <nnet-dir> <abs-path-to-bnfeadir>";
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
olddata=$2
nndir=$3
bnfeadir=$4

######## CONFIGURATION
norm_vars=$(cat $nndir/norm_vars)
splice_lr=$(cat $nndir/splice_lr)
feat_type=$(cat $nndir/feat_type)
cmvn_g=$nndir/cmvn_glob.mat

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $bnfeadir || exit 1;
mkdir -p $data || exit 1;

scp=$olddata/feats.scp
required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if [ ! -d $olddata/split$nj -o $olddata/split$nj -ot $olddata/feats.scp ]; then
  scripts/split_data.sh $olddata $nj
fi


#cut the MLP
nnet=$data/feature_extractor.nnet
nnet-trim-n-last-transforms --n=$trim_transforms --binary=false $nndir/final.nnet $nnet 2>${nnet}_log

#copy source data to new data dir....
cp $olddata/* $data 2>/dev/null; rm $data/feats.scp;

rm $data/.error 2>/dev/null

echo "Creating bnfeats into $data"


# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.
for ((n=0; n<nj; n++)); do
  log=$data/make_bnfeats.$n.log
  # prepare features : do per-speaker CMVN and splicing
  feats="ark:compute-cmvn-stats --spk2utt=ark:$olddata/split$nj/$n/spk2utt scp:$olddata/split$nj/$n/feats.scp ark:- | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$olddata/split$nj/$n/utt2spk ark:- scp:$olddata/split$nj/$n/feats.scp ark:- | splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
  # Choose further processing by feat_type
  case $feat_type in
    plain)
    ;;
    traps)
      transf=$nndir/hamm_dct.mat
      feats="$feats transform-feats --print-args=false $transf ark:- ark:- |"
    ;;
    transf)
      feats="$feats transform-feats $nndir/final.mat ark:- ark:- |"
    ;;
    transf-sat)
      echo yet unimplemented...
      exit 1;
    ;;
    *)
      echo "Unknown feature type $feat_type"
      exit 1;
  esac
  # Norm
  feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"

  # MLP forward 
  $cmd $log \
    nnet-forward $nnet "$feats" \
    ark,scp:$bnfeadir/raw_bnfea_$name.$n.ark,$bnfeadir/raw_bnfea_$name.$n.scp \
    || touch $data/.error &
 
done
wait;

N0=$(cat $olddata/feats.scp | wc -l) 
N1=$(cat $bnfeadir/raw_bnfea_$name.*.scp | wc -l)
if [[ -f $data/.error && "$N0" != "$N1" ]]; then
  echo "Error producing bnfea features for $name:"
  echo "Original feats : $N0  Bottleneck feats : $N1"
  tail $data/make_bnfeats.*.log.bak.1
  exit 1;
fi

if [[ -f $data/.error ]]; then
  echo "Warning : .error producing bnfea features, but all the $N1 features were computed...";
fi

# concatenate the .scp files together.
for ((n=0; n<nj; n++)); do
  cat $bnfeadir/raw_bnfea_$name.$n.scp >> $data/feats.scp
done


echo "Succeeded creating MLP-BN features for $name"

