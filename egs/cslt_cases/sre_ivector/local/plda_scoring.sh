#!/bin/bash
# Copyright 2015   David Snyder
#           2018   Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

# This script trains PLDA models and does scoring on i-vector or d-vector.

use_existing_models=false
simple_length_norm=true  # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-vec-dir> <enroll-vec-dir> <test-vec-dir> <trials-file> <scores-dir>"
fi

plda_data_dir=$1
enroll_data_dir=$2
test_data_dir=$3
plda_vec_dir=$4
enroll_vec_dir=$5
test_vec_dir=$6
trials=$7
scores_dir=$8

if [ "$use_existing_models" == "true" ]; then
  for f in ${plda_vec_dir}/mean.vec ${plda_vec_dir}/plda ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done
else
  run.pl ${plda_vec_dir}/log/compute_mean.log \
    ivector-normalize-length scp:${plda_vec_dir}/vector.scp \
    ark:- \| ivector-mean ark:- ${plda_vec_dir}/mean.vec || exit 1;

  run.pl $plda_vec_dir/log/plda.log \
    ivector-compute-plda ark:$plda_data_dir/spk2utt \
    "ark:ivector-normalize-length scp:${plda_vec_dir}/vector.scp  ark:- |" \
    $plda_vec_dir/plda || exit 1;
fi

mkdir -p $scores_dir/log

run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    --simple-length-normalization=$simple_length_norm \
    --num-utts=ark:${enroll_vec_dir}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${plda_vec_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${plda_vec_dir}/mean.vec scp:${enroll_vec_dir}/spk_vector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:${test_vec_dir}/vector.scp ark:- | ivector-subtract-global-mean ${plda_vec_dir}/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;
