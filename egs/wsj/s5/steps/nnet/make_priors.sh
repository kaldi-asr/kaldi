#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
use_gpu=no
ivector=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 2 ]; then
   echo "usage: $0 [options] <data-dir> <nnet-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
nndir=$2

######## CONFIGURATION

required="$data/feats.scp $nndir/final.nnet $nndir/final.feature_transform"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

echo "Accumulating prior stats by forwarding '$data' with '$nndir'"

# We estimate priors on 10k utterances, selected randomly from the splitted data,
N=$((10000/nj))

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
feats="ark:cat $sdata/JOB/feats.scp | utils/shuffle_list.pl --srand 777 | head -n$N | copy-feats scp:- ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  [ -z $ivector ] && echo "Missing --ivector, they were used in training!" && exit 1
  # Get the tool, 
  ivector_append_tool=append-vector-to-feats # default,
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  # Check dims,
  feats_job_1=$(sed 's:JOB:1:g' <(echo $feats))
  dim_raw=$(feat-to-dim "$feats_job_1" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats_job_1 $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  [ $dim_ivec != "$(cat $D/ivector_dim)" ] && \
    echo "Error, i-vector dim. mismatch (expected $(cat $D/ivector_dim), got $dim_ivec in '$ivector')" && \
    exit 1
  # Append to feats,
  feats="$feats $ivector_append_tool ark:- '$ivector' ark:- |"
fi

# Run the forward pass,
$cmd JOB=1:$nj $nndir/log/prior_stats.JOB.log \
  nnet-forward --use-gpu=$use_gpu --feature-transform=$nndir/final.feature_transform $nndir/final.nnet "$feats" ark:- \| \
  compute-cmvn-stats --binary=false ark:- $nndir/JOB.prior_cmvn_stats || exit 1

sum-matrices --binary=false $nndir/prior_cmvn_stats $nndir/*.prior_cmvn_stats 2>$nndir/log/prior_sum_matrices.log || exit 1
rm $nndir/*.prior_cmvn_stats

awk 'NR==2{ $NF=""; print "[",$0,"]"; }' $nndir/prior_cmvn_stats >$nndir/prior_counts || exit 1
    
echo "Succeeded creating prior counts '$nndir/prior_counts' from '$data'" 

