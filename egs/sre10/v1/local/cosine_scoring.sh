#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains an LDA transform and does cosine scoring.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 <enroll-data-dir> <test-data-dir> <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
fi

enroll_data_dir=$1
test_data_dir=$2
enroll_ivec_dir=$3
test_ivec_dir=$4
trials=$5
scores_dir=$6

mkdir -p $scores_dir
cat $trials | awk '{print $1, $2}' | \
 ivector-compute-dot-products - \
  scp:${enroll_ivec_dir}/spk_ivector.scp \
  scp:${test_ivec_dir}/ivector.scp \
   $scores_dir/cosine_scores
