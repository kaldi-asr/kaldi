#!/bin/bash 

# Copyright 2012-2014 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
remove_last_components=4 # remove N last components from the nnet
use_gpu=no
htk_save=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: $0 [options] <tgt-data-dir> <src-data-dir> <nnet-dir> <log-dir> <abs-path-to-bn-feat-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
nndir=$3
logdir=$4
bnfeadir=$5

######## CONFIGURATION

# copy the dataset metadata from srcdata.
mkdir -p $data $logdir $bnfeadir || exit 1;
utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null

# make $bnfeadir an absolute pathname.
[ '/' != ${bnfeadir:0:1} ] && bnfeadir=$PWD/$bnfeadir

required="$srcdata/feats.scp $nndir/final.nnet $nndir/final.feature_transform"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Concat feature transform with trimmed MLP:
nnet=$bnfeadir/feature_extractor.nnet
nnet-concat $nndir/final.feature_transform "nnet-copy --remove-last-components=$remove_last_components $nndir/final.nnet - |" $nnet 2>$logdir/feature_extractor.log || exit 1
nnet-info $nnet >$data/feature_extractor.nnet-info

echo "Creating bn-feats into $data"

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nndir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
#

if [ $htk_save == false ]; then
  # Run the forward pass,
  $cmd JOB=1:$nj $logdir/make_bnfeats.JOB.log \
    nnet-forward --use-gpu=$use_gpu $nnet "$feats" \
    ark,scp:$bnfeadir/raw_bnfea_$name.JOB.ark,$bnfeadir/raw_bnfea_$name.JOB.scp \
    || exit 1;
  # concatenate the .scp files
  for ((n=1; n<=nj; n++)); do
    cat $bnfeadir/raw_bnfea_$name.$n.scp >> $data/feats.scp
  done

  # check sentence counts,
  N0=$(cat $srcdata/feats.scp | wc -l) 
  N1=$(cat $data/feats.scp | wc -l)
  [[ "$N0" != "$N1" ]] && echo "$0: sentence-count mismatch, $srcdata $N0, $data $N1" && exit 1
  echo "Succeeded creating MLP-BN features '$data'"

else # htk_save == true
  # Run the forward pass saving HTK features,
  $cmd JOB=1:$nj $logdir/make_bnfeats_htk.JOB.log \
    mkdir -p $data/htkfeats/JOB \; \
    nnet-forward --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
    copy-feats-to-htk --output-dir=$data/htkfeats/JOB ark:- || exit 1
  # Make list of htk features,
  find $data/htkfeats -name *.fea >$data/htkfeats.scp

  # Check sentence counts,
  N0=$(cat $srcdata/feats.scp | wc -l)
  N1=$(find $data/htkfeats.scp | wc -l)
  [[ "$N0" != "$N1" ]] && echo "$0: sentence-count mismatch, $srcdata $N0, $data/htk* $N1" && exit 1
  echo "Succeeded creating MLP-BN features '$data/htkfeats.scp'"
fi
