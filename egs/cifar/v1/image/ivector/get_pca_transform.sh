#!/bin/bash

# Copyright 2017  David Snyder
#
# This script computes a PCA transform on top of image patches
#
#
# Apache 2.0.

# Begin configuration.
cmd=run.pl
config=
stage=0
dim=40 # The dim after applying PCA
normalize_variance=true # If the PCA transform normalizes the variance
normalize_mean=true # If the PCA transform centers
patch_width=8
patch_height=8
max_images=5000 # maximum number of files to use
subsample=5 # subsample features with this periodicity

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: steps/nnet2/get_pca_transform.sh [options] <data> <dir>"
  echo " e.g.: steps/train_pca_transform.sh data/train_si84 exp/tri2b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

data=$1
dir=$2

for f in $data/images.scp ; do
  [ ! -f "$f" ] && echo "$0: expecting file $f to exist" && exit 1
done

patch_opts="--patch-width=$patch_width --patch-height=$patch_height"

mkdir -p $dir/log
echo $patch_opts > $dir/patch_opts

feats="ark,s,cs:utils/subset_scp.pl --quiet $max_images $data/images.scp | extract-patches $patch_opts scp:- ark:- | subsample-feats --n=$subsample ark:- ark:- |"

if [ $stage -le 0 ]; then
  $cmd $dir/log/pca_est.log \
    est-pca --dim=$dim --normalize-variance=$normalize_variance \
    --normalize-mean=$normalize_mean "$feats" $dir/final.mat || exit 1;
fi

echo "Done estimating PCA transform in $dir"

exit 0
