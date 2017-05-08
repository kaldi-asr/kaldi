#!/bin/bash

# Copyright     2013  Daniel Povey
#          2014-2017  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of images, given
# features and a trained iVector extractor.

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


if [ $# != 3 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor_2048 data/train exp/ivectors"
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

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.ubm $srcdir/final.ie $data/images.scp $srcdir/final.mat $srcdir/patch_opts; do
   [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1
done

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
image/split_image_dir.sh $data $nj || exit 1;

patch_opts=$(cat $srcdir/patch_opts)

## Set up features.
feats="ark,s,cs:extract-patches $patch_opts scp:$sdata/JOB/images.scp ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  dubm="fgmm-global-to-gmm $srcdir/final.ubm -|"

  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $srcdir/final.ubm "$feats" \
       ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract --verbose=2 $srcdir/final.ie "$feats" ark,s,cs:- \
      ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
fi
