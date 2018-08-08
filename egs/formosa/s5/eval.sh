#!/bin/bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#
# Before you run this recips, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1-Eval").
# For more detail, please check:
# 1. Formosa Speech in the Wild (FSW) project (https://sites.google.com/speech.ntut.edu.tw/fsw/home/corpus)
# 2. Formosa Speech Recognition Challenge (FSW) 2018 (https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge)
stage=-2
train_stage=-10
num_jobs=20

# shell options
set -e -o pipefail

. ./cmd.sh
. ./utils/parse_options.sh

# configure number of jobs running in parallel, you should adjust these numbers according to your machines
# data preparation
if [ $stage -le -2 ]; then

  # Data Preparation
  echo "$0: Data Preparation"
  local/prepare_eval_data.sh || exit 1;

fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

# mfcc
if [ $stage -le -1 ]; then

  echo "$0: making mfccs"
  for x in eval; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $num_jobs data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done

fi

# mono
if [ $stage -le 0 ]; then

  # Monophone decoding
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/mono/graph data/eval exp/mono/decode_eval
  )

fi

# tri1
if [ $stage -le 1 ]; then

  # decode tri1
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri1/graph data/eval exp/tri1/decode_eval
  )

fi

# tri2
if [ $stage -le 2 ]; then

  # decode tri2
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri2/graph data/eval exp/tri2/decode_eval
  )

fi

# tri3a
if [ $stage -le 3 ]; then

  # decode tri3a
  (
  steps/decode.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri3a/graph data/eval exp/tri3a/decode_eval
  )

fi

# tri4
if [ $stage -le 4 ]; then

  # decode tri4a
  (
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri4a/graph data/eval exp/tri4a/decode_eval
  )

fi

# tri5
if [ $stage -le 5 ]; then

  # decode tri5
  (
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
     exp/tri5a/graph data/eval exp/tri5a/decode_eval || exit 1;
  )

fi

exit 0;

# nnet3 tdnn models
if [ $stage -le 6 ]; then

  train_stage=99
  echo "$0: evaluate nnet3 model"
  local/nnet3/run_tdnn.sh --stage $train_stage

fi

# chain model
if [ $stage -le 7 ]; then

  train_stage=99
  echo "$0: evaluate chain model"
  local/chain/run_tdnn.sh --stage $train_stage

fi

# getting results (see RESULTS file)
if [ $stage -le 10 ]; then

  echo "$0: extract the results"
  rm -f eval-decoding-results.log
  touch eval-decoding-results.log
  for x in exp/*/decode_eval/log; do [ -d $x ] && grep NER $x/*.log | grep -v LOG >> eval-decoding-results.log ; done 2>/dev/null
  for x in exp/*/*/decode_eval/log; do [ -d $x ] && grep WER $x/*.log | grep -v LOG >> eval-decoding-results.log; done 2>/dev/null

fi

# finish
echo "$0: all done"

exit 0;
