#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.5
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 1 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor data/train exp/ivectors"
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

dir=$1

mkdir -p $dir/tmp

for f in $dir/scores.scp $dir/spk2utt $dir/utt2spk $dir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $dir/spk2utt $dir/tmp/
cp $dir/utt2spk $dir/tmp/
cp $dir/segments $dir/tmp/

utils/fix_data_dir.sh $dir/tmp > /dev/null

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="utils/filter_scp.pl $sdata/JOB/spk2utt $dir/scores.scp |"
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
    agglomerative-cluster --threshold=$threshold \
      scp:"$feats" ark,t:$sdata/JOB/spk2utt \
      ark,t:$dir/labels.JOB.txt || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining "
  for j in $(seq $nj); do cat $dir/labels.$j.txt; done >$dir/labels.txt || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi
