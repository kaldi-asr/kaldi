#!/bin/bash

# Copyright     2015  Vimal Manohar
# Apache 2.0.

# This script creates segment-level ivectors from recording-level ivectors.

# Begin configuration section.
cmd="run.pl"
stage=0
ivector_period=10
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data> <recording-ivector-dir> <ivector-dir>"
  echo " e.g.: $0 data/test exp/nnet2_online/ivectors_test_reco exp/nnet2_online/ivectors_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  exit 1;
fi

data=$1
reco_ivector_dir=$2
dir=$3

for f in $data/feats.scp $reco_ivector_dir/ivectors_seg.ark $reco_ivector_dir/reco_segmentation.ark; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $dir/log

echo $ivector_period > $dir/ivector_period || exit 1;

nj=$(cat $reco_ivector_dir/num_jobs) || exit 1

# This will probably work fine because both have the same number of recordings;
# but needs to be checked. Otherwise old data dir must also be input.
utils/split_data.sh --per-reco $data $nj
sdata=$data/split$nj

if [ $stage -le 0 ]; then
  if [ -f $data/segments ]; then
    $cmd JOB=1:$nj $dir/log/get_ivectors_utt.JOB.log \
      ivector-split-to-segments --n=$ivector_period $sdata/JOB/segments \
      ark:$reco_ivector_dir/reco_segmentation.JOB.ark \
      ark:$reco_ivector_dir/ivectors_reco.JOB.ark ark:- \| append-feats \
      "ark:subsample-feats --n=$ivector_period scp:$sdata/JOB/feats.scp |" \
      ark:- ark,scp:$dir/ivector_online.JOB.ark,$dir/ivector_online.JOB.scp || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/get_ivectors_utt.JOB.log \
      ivector-split-to-segments --n=$ivector_period \
      ark:$reco_ivector_dir/reco_segmentation.JOB.ark \
      ark:$reco_ivector_dir/ivectors_reco.JOB.ark ark:- \| append-feats \
      "ark:subsample-feats --n=$ivector_period scp:$sdata/JOB/feats.scp |" \
      ark:- ark,scp:$dir/ivector_online.JOB.ark,$dir/ivector_online.JOB.scp || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi

echo "$0: done extracting (pseudo-online) iVectors"
