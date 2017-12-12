#!/bin/bash

# Copyright  2016  David Snyder
#            2017  Matthew Maciejewski
# Apache 2.0.

# This script computes the stopping threshold used in clustering. This is done
# by using k-means clustering with a k of 2 on the PLDA scores for each
# recording. The final threshold is the average of the midpoints between the
# means of the two clusters.

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


if [ $# != 3 ]; then
  echo "Usage: $0 <plda-dir> <ivector-dir> <output-dir>"
  echo " e.g.: $0 exp/ivectors_callhome_heldout exp/ivectors_callhome_test exp/ivectors_callhome_test"
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

pldadir=$1
ivecdir=$2
dir=$3

mkdir -p $dir/tmp

for f in $ivecdir/ivector.scp $ivecdir/spk2utt $ivecdir/utt2spk $ivecdir/segments $pldadir/plda $pldadir/mean.vec $pldadir/transform.mat; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
cp $ivecdir/ivector.scp $dir/tmp/feats.scp
cp $ivecdir/spk2utt $dir/tmp/
cp $ivecdir/utt2spk $dir/tmp/
cp $ivecdir/segments $dir/tmp/

utils/fix_data_dir.sh $dir/tmp > /dev/null

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="ark:ivector-subtract-global-mean $pldadir/mean.vec scp:$sdata/JOB/feats.scp ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
if [ $stage -le 0 ]; then
  echo "$0: Computing calibration thresholds"
  $cmd JOB=1:$nj $dir/log/compute_calibration.JOB.log \
    ivector-plda-scoring-dense --target-energy=$target_energy $pldadir/plda \
      ark:$sdata/JOB/spk2utt "$feats" ark:- \
      \| compute-calibration --spk2utt-rspecifier=ark:$sdata/JOB/spk2utt \
      ark:- ark,t:$dir/threshold.JOB.txt || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining calibration thresholds across jobs"
  for j in $(seq $nj); do cat $dir/threshold.$j.txt; echo; done | sed '/^$/d' >$dir/thresholds.txt || exit 1;
  awk '{ sum += $NF; n++ } END { if (n > 0) print sum / n; }' $dir/thresholds.txt > $dir/threshold.txt
fi

if $cleanup ; then
  rm -rf $dir/tmp
  for j in $(seq $nj); do rm $dir/threshold.$j.txt; done || exit 1;
  rm $dir/thresholds.txt || exit 1;
fi
