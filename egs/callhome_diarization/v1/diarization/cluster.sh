#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.

# TODO This script performs agglomerative clustering.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.5
utt2num=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --target-energy <target-energy|0.1>              # Target energy remaining in iVectors after applying"
  echo "                                                   # a conversation dependent PCA."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

srcdir=$1
dir=$2

mkdir -p $dir/tmp

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/
utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z $utt2num ]; then
  utt2num="ark,t:$utt2num"
fi

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="utils/filter_scp.pl $sdata/JOB/spk2utt $srcdir/scores.scp |"
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
    agglomerative-cluster --threshold=$threshold \
      --utt2num-rspecifier=$utt2num scp:"$feats" \
      ark,t:$sdata/JOB/spk2utt ark,t:$dir/labels.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  python diarization/make_rttm.py $srcdir/segments $dir/labels > $dir/rttm || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi
