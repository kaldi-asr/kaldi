#!/bin/bash 

# Copyright 2012  Karel Vesely, Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
trim_transforms=4
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: $0 [oprtions] <tgt-data-dir> <src-data-dir> <nnet-dir> <log-dir> <abs-path-to-bn-feat-dir>";
   echo "options: "
   echo "  --trim-transforms <N>                            # number of NNet Components to remove from the end"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
nndir=$3
logdir=$4
bnfeadir=$5

######## CONFIGURATION
norm_vars=$(cat $nndir/norm_vars)
feat_type=$(cat $nndir/feat_type)
cmvn_g=$nndir/cmvn_glob.mat

# copy the dataset metadata from srcdata.
mkdir -p $data || exit 1;
cp $srcdata/* $data 2>/dev/null; rm $data/feats.scp $data/cmvn.scp;

# make $bnfeadir an absolute pathname.
bnfeadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $bnfeadir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $bnfeadir || exit 1;
mkdir -p $data || exit 1;
mkdir -p $logdir || exit 1;


srcscp=$srcdata/feats.scp
scp=$data/feats.scp

required="$srcscp $nndir/final.nnet $cmvn_g $srcdata/cmvn.scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if [ ! -d $srcdata/split$nj -o $srcdata/split$nj -ot $srcdata/feats.scp ]; then
  utils/split_data.sh $srcdata $nj
fi


#cut the MLP
nnet=$bnfeadir/feature_extractor.nnet
nnet-trim-n-last-transforms --n=$trim_transforms --binary=false $nndir/final.nnet $nnet 2>$logdir/feature_extractor.log

#get the feature transform
feature_transform=$nndir/$(readlink $nndir/final.feature_transform)


rm $data/.error 2>/dev/null

echo "Creating bn-feats into $data"


# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.
for ((n=1; n<=nj; n++)); do
  log=$logdir/make_bnfeats.$n.log
  # Prepare feature pipeline
  feats="ark,s,cs:copy-feats scp:$srcdata/cmvn.scp ark:- |"
  # Optionally add cmvn
  if [ -f $nndir/norm_vars ]; then
    norm_vars=$(cat $nndir/norm_vars 2>/dev/null)
    feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
  fi
  # Optionally add deltas
  if [ -f $nndir/delta_order ]; then
    delta_order=$(cat $nndir/delta_order)
    feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
  fi

  # MLP forward (with feature transform) 
  $cmd $log \
    nnet-forward --feature-transform=$feature_transform $nnet "$feats" \
    ark,scp:$bnfeadir/raw_bnfea_$name.$n.ark,$bnfeadir/raw_bnfea_$name.$n.scp \
    || touch $data/.error &
 
done
wait;

N0=$(cat $srcdata/feats.scp | wc -l) 
N1=$(cat $bnfeadir/raw_bnfea_$name.*.scp | wc -l)
if [[ -f $data/.error && "$N0" != "$N1" ]]; then
  echo "Error producing bnfea features for $name:"
  echo "Original feats : $N0  Bottleneck feats : $N1"
  exit 1;
fi

if [[ -f $data/.error ]]; then
  echo "Warning : .error producing bnfea features, but all the $N1 features were computed...";
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $bnfeadir/raw_bnfea_$name.$n.scp >> $data/feats.scp
done


echo "Succeeded creating MLP-BN features for $name ($data)"

