#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.

# TODO This script computes the stopping threshold used in clustering.

# Begin configuration section.
cmd="run.pl"
stage=0
target_energy=0.1
nj=10
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 1 ]; then
  echo "Usage: $0 <ivector-dir>"
  echo " e.g.: $0 exp/ivectors"
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

for f in $dir/ivector.scp $dir/spk2utt $dir/utt2spk $dir/segments $dir/plda $dir/mean.vec $dir/transform.mat; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
cp $dir/ivector.scp $dir/tmp/feats.scp
cp $dir/spk2utt $dir/tmp/
cp $dir/utt2spk $dir/tmp/
cp $dir/segments $dir/tmp/

utils/fix_data_dir.sh $dir/tmp > /dev/null

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="ark:ivector-subtract-global-mean $dir/mean.vec scp:$sdata/JOB/feats.scp ark:- | transform-vec $dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
if [ $stage -le 0 ]; then
  echo "$0: Computing calibration thresholds"
  $cmd JOB=1:$nj $dir/log/compute_calibration.JOB.log \
    ivector-plda-scoring-dense --target-energy=$target_energy $dir/plda \
      ark:$sdata/JOB/spk2utt "$feats" ark:- \
      \| compute-calibration ark:- $dir/threshold.JOB.txt || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining calibration thresholds across jobs"
  for j in $(seq $nj); do cat $dir/threshold.$j.txt; echo; done >$dir/thresholds.txt || exit 1;
  awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }' $dir/thresholds.txt > $dir/threshold.txt
fi

if $cleanup ; then
  rm -rf $dir/tmp
  for j in $(seq $nj); do rm $dir/threshold.$j.txt; done || exit 1;
  rm $dir/thresholds.txt || exit 1;
fi
