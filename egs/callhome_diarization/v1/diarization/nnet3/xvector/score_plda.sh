#!/usr/bin/env bash
# Copyright  2016-2018  David Snyder
#            2017-2018  Matthew Maciejewski
# Apache 2.0.

# This script is a modified version of diarization/score_plda.sh
# that replaces i-vectors with x-vectors.
#
# This script computes PLDA scores from pairs of x-vectors extracted
# from segments of a recording.  These scores are in the form of
# affinity matrices, one for each recording.  Most likely, the x-vectors
# were computed using diarization/nnet3/xvector/extract_xvectors.sh.
# The affinity matrices are most likely going to be clustered using
# diarization/cluster.sh.

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
  echo "Usage: $0 <plda-dir> <xvector-dir> <output-dir>"
  echo " e.g.: $0 exp/xvectors_callhome_heldout exp/xvectors_callhome_test exp/xvectors_callhome_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --target-energy <target-energy|0.1>              # Target energy remaining in xvectors after applying"
  echo "                                                   # a conversation dependent PCA."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

pldadir=$1
xvecdir=$2
dir=$3

mkdir -p $dir/tmp

for f in $xvecdir/xvector.scp $xvecdir/spk2utt $xvecdir/utt2spk $xvecdir/segments $pldadir/plda $pldadir/mean.vec $pldadir/transform.mat; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
cp $xvecdir/xvector.scp $dir/tmp/feats.scp
cp $xvecdir/spk2utt $dir/tmp/
cp $xvecdir/utt2spk $dir/tmp/
cp $xvecdir/segments $dir/tmp/
cp $xvecdir/spk2utt $dir/
cp $xvecdir/utt2spk $dir/
cp $xvecdir/segments $dir/

utils/fix_data_dir.sh $dir/tmp > /dev/null

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="ark:ivector-subtract-global-mean $pldadir/mean.vec scp:$sdata/JOB/feats.scp ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
if [ $stage -le 0 ]; then
  echo "$0: scoring xvectors"
  $cmd JOB=1:$nj $dir/log/plda_scoring.JOB.log \
    ivector-plda-scoring-dense --target-energy=$target_energy $pldadir/plda \
      ark:$sdata/JOB/spk2utt "$feats" ark,scp:$dir/scores.JOB.ark,$dir/scores.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining PLDA scores across jobs"
  for j in $(seq $nj); do cat $dir/scores.$j.scp; done >$dir/scores.scp || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi
