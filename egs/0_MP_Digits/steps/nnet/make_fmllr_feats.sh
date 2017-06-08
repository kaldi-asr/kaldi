#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely),
#                 
# Apache 2.0.
#
# This script dumps fMLLR features in a new data directory, 
# which is later used for neural network training/testing.

# Begin configuration section.  
nj=4
cmd=run.pl
transform_dir=
raw_transform_dir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <tgt-data-dir> <src-data-dir> <gmm-dir> <log-dir> <fea-dir>"
   echo "e.g.: $0 data-fmllr/train data/train exp/tri5a exp/make_fmllr_feats/log plp/processed/"
   echo ""
   echo "This script dumps fMLLR features to disk, so it can be used for NN training."
   echo "It automoatically figures out the 'feature-type' of the source GMM systems."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --transform-dir <transform-dir>                  # dir with fMLLR transforms"
   echo "  --raw-transform-dir <transform-dir>              # dir with raw-fMLLR transforms"
   exit 1;
fi

data=$1
srcdata=$2
gmmdir=$3
logdir=$4
feadir=$5

sdata=$srcdata/split$nj;

# Get the config,
D=$gmmdir
[ -f $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts) || cmvn_opts=
[ -f $D/delta_opts ] && delta_opts=$(cat $D/delta_opts) || delta_opts=
[ -f $D/splice_opts ] && splice_opts=$(cat $D/splice_opts) || splice_opts=

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Check files exist,
for f in $sdata/1/feats.scp $sdata/1/cmvn.scp; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done
[ ! -z "$transform_dir" -a ! -f $transform_dir/trans.1 ] && \
  echo "$0: Missing $transform_dir/trans.1" && exit 1;
[ ! -z "$raw_transform_dir" -a ! -f $raw_transform_dir/raw_trans.1 ] && \
  echo "$0: Missing $raw_transform_dir/raw_trans.1" && exit 1;

# Figure-out the feature-type,
feat_type="[UNKNOWN]"
[ -z "$raw_transform_dir" -a ! -f $gmmdir/final.mat -a ! -z "$transform_dir" ] && feat_type=delta_fmllr
[ -z "$raw_transform_dir" -a -f $gmmdir/final.mat -a ! -z "$transform_dir" ] && feat_type=lda_fmllr
[ ! -z "$raw_transform_dir" ] && feat_type=raw_fmllr
echo "$0: feature type is $feat_type";

# Hand-code the feature pipeline,
case $feat_type in
  delta_fmllr) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk \"ark:cat $transform_dir/trans.* |\" ark:- ark:- |";;
  lda_fmllr) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $gmmdir/final.mat ark:- ark:- | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk \"ark:cat $transform_dir/trans.* |\" ark:- ark:- |";;
  raw_fmllr) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$raw_transform_dir/raw_trans.JOB ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

# Prepare the output dir,
utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null
# Make $feadir an absolute pathname,
[ '/' != ${feadir:0:1} ] && feadir=$PWD/$feadir

# Store the output-features,
name=`basename $data`
$cmd JOB=1:$nj $logdir/make_fmllr_feats.JOB.log \
  copy-feats "$feats" \
  ark,scp:$feadir/feats_fmllr_$name.JOB.ark,$feadir/feats_fmllr_$name.JOB.scp || exit 1;
   
# Merge the scp,
for n in $(seq 1 $nj); do
  cat $feadir/feats_fmllr_$name.$n.scp 
done > $data/feats.scp

echo "$0: Done!, type $feat_type, $srcdata --> $data, using : raw-trans ${raw_transform_dir:-None}, gmm $gmmdir, trans ${transform_dir:-None}"

exit 0;
