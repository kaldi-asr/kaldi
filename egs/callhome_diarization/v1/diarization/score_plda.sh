#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

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


if [ $# != 2 ]; then
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

pldadir=$1
dir=$2

cat $nj > $dir/num_jobs

mkdir -p $dir/tmp

for f in $dir/ivector.scp $dir/spk2utt $dir/utt2spk $dir/segments $pldadir/plda $pldadir/mean.vec $pldadir/transform.mat; do
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

feats="ark:ivector-subtract-global-mean $pldadir/mean.vec scp:$sdata/JOB/feats.scp ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
if [ $stage -le 0 ]; then
  echo "$0: scoring iVectors"
  $cmd JOB=1:$nj $dir/log/plda_scoring.JOB.log \
    ivector-plda-scoring-dense --target-energy=$target_energy $pldadir/plda \
      ark:$sdata/JOB/spk2utt "$feats" ark,scp:$dir/scores.JOB.ark,$dir/scores.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining calibration thresholds across jobs"
  for j in $(seq $nj); do cat $dir/scores.$j.scp; done >$dir/scores.scp || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi
