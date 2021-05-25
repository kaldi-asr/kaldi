#!/bin/bash

# Copyright       2016  David Snyder
#            2017-2018  Matthew Maciejewski
#                 2020  Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

# This script performs spectral clustering using scored
# pairs of subsegments and produces a rttm file with speaker
# labels derived from the clusters.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
rttm_channel=0
min_neighbors=3 # This is the min "p" value of number of neighbors to start
                # thresholding the binary matrix. In the original paper, this
                # is set to 2. Thanks to Hassan Taherian for pointing out that
                # this value leads to instability since it often causes the
                # affinity to not be fully connected. Hence we set it to 3 here
                # by default.
reco2num_spk=
rttm_affix=

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
  echo "  --min-neighbors <p|3>                            # Min value for affinity matrix thresholding"
  echo "  --rttm-channel <rttm-channel|0>                  # The value passed into the RTTM channel field. Only affects"
  echo "                                                   # the format of the RTTM file."
  echo "  --reco2num-spk <reco2num-spk-file>               # File containing mapping of recording ID"
  echo "                                                   # to number of speakers. Used instead of threshold"
  echo "                                                   # as stopping criterion if supplied."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

srcdir=$1
dir=$2

reco2num_spk_opts=
if [ ! $reco2num_spk == "" ]; then
  reco2num_spk_opts="--reco2num_spk $reco2num_spk"
fi

mkdir -p $dir/tmp

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/
utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z $reco2num_spk ]; then
  reco2num_spk="ark,t:$reco2num_spk"
fi

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="utils/filter_scp.pl $sdata/JOB/spk2utt $srcdir/scores.scp |"
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  for j in `seq $nj`; do 
    utils/filter_scp.pl $sdata/$j/spk2utt $srcdir/scores.scp > $dir/scores.$j.scp
  done
  $cmd JOB=1:$nj $dir/log/spectral_cluster.JOB.log \
    python3 diarization/spec_clust.py $reco2num_spk_opts --min_neighbors $min_neighbors \
      scp:$dir/scores.JOB.scp ark,t:$sdata/JOB/spk2utt ark,t:$dir/labels.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  diarization/make_rttm.py --rttm-channel $rttm_channel $srcdir/segments $dir/labels $dir/rttm${rttm_affix} || exit 1;
fi

if $cleanup ; then
  rm -r $dir/tmp || exit 1;
fi
