#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
stage=0

## DNN opts
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <data-dir> <multi-stream-opts> <nnet-dir> <mdelta-stats-dir> <log-dir> <out-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

srcdata=$1
multi_stream_opts=$2
nnet_dir=$3
mdelta_stats_dir=$4
logdir=$5
outdir=$6

outdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $outdir ${PWD}`

# Check required 
required="$srcdata/feats.scp"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

mkdir -p $logdir
mkdir -p $outdir 

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

nnet=$nnet_dir/final.nnet
feature_transform=$nnet_dir/final.feature_transform
# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nnet_dir
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
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# Add Multi-stream options
feats="$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask $multi_stream_opts ark:- ark:- |"

# transf_nnet_out_opts
transf_nnet_out_opts="--pdf-to-pseudo-phone=$mdelta_stats_dir/pdf_to_pseudo_phone.txt"

if [ $stage -le 0 ]; then
$cmd JOB=1:$nj $logdir/compute_mdelta_scores.JOB.log \
  tmp_dir=\$\(mktemp -d\) '&&' mkdir -p \$tmp_dir '&&' \
  nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
  transform-nnet-posteriors $transf_nnet_out_opts ark:- ark,scp:\$tmp_dir/post.JOB.ark,\$tmp_dir/post.JOB.scp '&&' \
  python utils/multi-stream/pm_utils/compute_mdelta_post.py \$tmp_dir/post.JOB.scp $mdelta_stats_dir/pri $outdir/mdelta_scores.JOB.pklz '&&' \
  rm -rf \$tmp_dir/ || exit 1;  
fi

# combine the splits
comb_str=""
for ((n=1; n<=nj; n++)); do
  comb_str=$comb_str" "$outdir/mdelta_scores.${n}.pklz
done

python utils/multi-stream/pm_utils/merge_dicts.py $comb_str $outdir/mdelta_scores.$name.pklz 2>$logdir/merge_dicts.log || exit 1;

exit 0;

