#!/usr/bin/env bash

# Copyright 2019 Junbo Zhang
# Apache 2.0

# This script shows how to calculate Goodness of Pronunciation (GOP) and
# extract phone-level pronunciation feature for mispronunciations detection
# tasks. Read ../README.md or the following paper for details:
#
# "Hu et al., Improved mispronunciation detection with deep neural network
# trained acoustic models and transfer learning based logistic regression
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
stage=0
nj=4

data=test_10short
dir=exp/gop_$data

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
  # Prepare test data
  [ -d data ] || mkdir -p data/$data
  local/make_testcase.sh $test_data data/$data
fi

if [ $stage -le 1 ]; then
  # Compute Log-likelihoods
  steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
    --online-ivector-dir $ivector data/$data $model exp/probs_$data
fi

if [ $stage -le 2 ]; then
  steps/nnet3/align.sh --cmd "$cmd" --nj $nj --use_gpu false \
    --online_ivector_dir $ivector data/$data $lang $model $dir
fi

if [ $stage -le 3 ]; then
  # make a map which converts phones to "pure-phones"
  # "pure-phone" means the phone whose stress and pos-in-word markers are ignored
  # eg. AE1_B --> AE, EH2_S --> EH, SIL --> SIL
  local/remove_phone_markers.pl $lang/phones.txt $dir/phones-pure.txt \
    $dir/phone-to-pure-phone.int

  # Convert transition-id to pure-phone id
  $cmd JOB=1:$nj $dir/log/ali_to_phones.JOB.log \
    ali-to-phones --per-frame=true $model/final.mdl "ark,t:gunzip -c $dir/ali.JOB.gz|" \
      "ark,t:-" \| utils/apply_map.pl -f 2- $dir/phone-to-pure-phone.int \| \
      gzip -c \>$dir/ali-pure-phone.JOB.gz   || exit 1;
fi

if [ $stage -le 4 ]; then
  # The outputs of the binary compute-gop are the GOPs and the phone-level features.
  #
  # An example of the GOP result (extracted from "ark,t:$dir/gop.3.txt"):
  # 4446-2273-0031 [ 1 0 ] [ 12 0 ] [ 27 -5.382001 ] [ 40 -13.91807 ] [ 1 -0.2555897 ] \
  #                [ 21 -0.2897284 ] [ 5 0 ] [ 31 0 ] [ 33 0 ] [ 3 -11.43557 ] [ 25 0 ] \
  #                [ 16 0 ] [ 30 -0.03224623 ] [ 5 0 ] [ 25 0 ] [ 33 0 ] [ 1 0 ]
  # It is in the posterior format, where each pair stands for [pure-phone-index gop-value].
  # For example, [ 27 -5.382001 ] means the GOP of the pure-phone 27 (it corresponds to the
  # phone "OW", according to "$dir/phones-pure.txt") is -5.382001, indicating the audio
  # segment of this phone should be a mispronunciation.
  #
  # The phone-level features are in matrix format:
  # 4446-2273-0031  [ -0.2462088 -10.20292 -11.35369 ...
  #                   -8.584108 -7.629755 -13.04877 ...
  #                   ...
  #                   ... ]
  # The row number is the phone number of the utterance. In this case, it is 17.
  # The column number is 2 * (pure-phone set size), as the feature is consist of LLR + LPR.
  # The phone-level features can be used to train a classifier with human labels. See Hu's
  # paper for detail.
  $cmd JOB=1:$nj $dir/log/compute_gop.JOB.log \
    compute-gop --phone-map=$dir/phone-to-pure-phone.int $model/final.mdl \
      "ark,t:gunzip -c $dir/ali-pure-phone.JOB.gz|" \
      "ark:exp/probs_$data/output.JOB.ark" \
      "ark,t:$dir/gop.JOB.txt" "ark,t:$dir/phonefeat.JOB.txt"   || exit 1;
  echo "Done compute-gop, the results: \"$dir/gop.<JOB>.txt\" in posterior format."

  # We set -5 as a universal empirical threshold here. You can also determine multiple phone
  # dependent thresholds based on the human-labeled mispronunciation data.
  echo "The phones whose gop values less than -5 could be treated as mispronunciations."
fi
