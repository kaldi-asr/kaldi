#!/bin/bash

# Copyright     2013  Daniel Povey
#               2014  David Snyder
#               2016  Matthew Maciejewski
# Apache 2.0.

# This script clusters iVectors for a set of utterances, given
# matrices of extracted iVectors and the number of speakers

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <data> <ivector-dir>"
  echo " e.g.: $0 data/train exp/ivectors"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  exit 1;
fi

data=$1
dir=$2

for f in $data/utt2num ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

if [ $stage -le 0 ]; then
  echo "$0: clustering iVectors"
  $cmd JOB=1:$nj $dir/log/cluster_ivectors.JOB.log \
    ivector-cluster --ivector-weights-rspecifier=scp:$dir/ivector_weights.JOB.scp \
      --utt2num-rxfilename=ark:$data/utt2num scp:$dir/ivector.JOB.scp \
      scp:$dir/ivector_ranges.JOB.scp \
      ark,scp,t:$dir/cluster_ranges.JOB.ark,$dir/cluster_ranges.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining cluster ranges across jobs"
  for j in $(seq $nj); do cat $dir/cluster_ranges.$j.scp; done >$dir/cluster_ranges.scp || exit 1;
fi
