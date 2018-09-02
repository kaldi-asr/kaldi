#!/bin/bash
# Copyright  2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Danwei Cai)
#            2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Ming Li)
# Apache 2.0.
#
# Train a PCA transform matrix with PPP 

# Begin configuration.
cmd=run.pl
config=
dim=52 # The dim after applying PCA
normalize_variance=true # If the PCA transform normalizes the variance
normalize_mean=true # If the PCA transform centers

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

data=$1
dir=$2

for f in $data/ppp.scp ; do
  [ ! -f "$f" ] && echo "$0: expecting file $f to exist" && exit 1
done

mkdir -p $dir/log

$cmd $dir/log/pca_est.log \
  est-pca --dim=$dim --normalize-variance=$normalize_variance \
  --normalize-mean=$normalize_mean scp:${data}/ppp.scp $dir/final.mat || exit 1;

echo "Done estimating PCA transform in $dir"

exit 0

