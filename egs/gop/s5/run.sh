#!/bin/bash

# Copyright 2019 Junbo Zhang
# Apache 2.0

# This script shows how to calculate Goodness of Pronunciation (GOP) and
# extract phone-level pronunciation feature for  mispronunciations detection
# tasks. Read ../README.md or the following paper for details:
#
# "Hu et al., Improved mispronunciation detection with deep neural network
# trained acoustic  models and transfer learning based logistic regression
# classifiers, 2015."

# You might not want to do this for interactive shells.
set -e

# Before running this recipe, you have to run the librispeech recipe firstly.
# This script assumes the following paths exist.
librispeech_eg=../../librispeech/s5
model=$librispeech_eg/exp/nnet3_cleaned/tdnn_sp
ivector=$librispeech_eg/exp/nnet3_cleaned/ivectors_test_clean_hires
lang=$librispeech_eg/data/lang
test_data=$librispeech_eg/data/test_clean_hires

for d in $model $ivector $lang $test_data; do
  [ ! -d $d ] && echo "$0: no such path $d" && exit 1;
done

# Global configurations
stage=10
nj=4

data=test_10short
dir=exp/gop_$data

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 10 ]; then
  # Prepare test data
  [ -d data ] || mkdir -p data/$data
  local/make_testcase.sh $test_data data/$data
fi

if [ $stage -le 20 ]; then
  # Compute Log-likelihoods
  steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
    --online-ivector-dir $ivector data/$data $model exp/probs_$data
fi

if [ $stage -le 30 ]; then
  steps/nnet3/align.sh --cmd "$cmd" --nj $nj --use_gpu false \
    --online_ivector_dir $ivector data/$data $lang $model $dir
fi

if [ $stage -le 40 ]; then
  # make a map which converts phones to "pure-phones"
  # "pure-phone" means the phone whose stress and pos-in-word symbols are ignored
  # eg. AE1_B --> AE, EH2_S --> EH, SIL --> SIL
  utils/remove_symbols_from_phones.pl $lang/phones.txt $dir/phones-pure.txt \
    $dir/phone-to-pure-phone.int

  # Convert transition-id to pure-phone id
  $cmd JOB=1:$nj $dir/log/ali_to_phones.JOB.log \
    ali-to-phones --per-frame=true $model/final.mdl "ark,t:gunzip -c $dir/ali.JOB.gz|" \
      "ark,t:-" \| utils/apply_map.pl -f 2- $dir/phone-to-pure-phone.int \| \
      gzip -c \>$dir/ali-pure-phone.JOB.gz   || exit 1;
fi

if [ $stage -le 50 ]; then
  # Compute GOP and phone-level feature
  $cmd JOB=1:$nj $dir/log/compute_gop.JOB.log \
    compute-gop --phone-map=$dir/phone-to-pure-phone.int $model/final.mdl \
      "ark,t:gunzip -c $dir/ali-pure-phone.JOB.gz|" \
      "ark:exp/probs_$data/output.JOB.ark" \
      "ark,t:$dir/gop.JOB.txt" "ark,t:$dir/phonefeat.JOB.txt"   || exit 1;

  echo "Done compute-gop, the results: \"$dir/gop.<JOB>.txt\" in posterior format."
  echo "The phones whose gop values less than -5 could be treated as mispronunciations."
fi
