#!/usr/bin/env bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#
# Before you run this recipe, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1").
# For more detail, please check:
# 1. Formosa Speech in the Wild (FSW) project (https://sites.google.com/speech.ntut.edu.tw/fsw/home/corpus)
# 2. Formosa Speech Recognition Challenge (FSW) 2018 (https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge)
stage=-2
num_jobs=20

train_dir=NER-Trs-Vol1/Train
eval_dir=NER-Trs-Vol1-Eval
eval_key_dir=NER-Trs-Vol1-Eval-Key

# shell options
set -eo pipefail

. ./cmd.sh
. ./utils/parse_options.sh

# configure number of jobs running in parallel, you should adjust these numbers according to your machines
# data preparation
if [ $stage -le -2 ]; then
  # Lexicon Preparation,
  echo "$0: Lexicon Preparation"
  local/prepare_dict.sh || exit 1;

  # Data Preparation
  echo "$0: Data Preparation"
  local/prepare_data.sh --train-dir $train_dir --eval-dir $eval_dir --eval-key-dir $eval_key_dir || exit 1;

  # Phone Sets, questions, L compilation
  echo "$0: Phone Sets, questions, L compilation Preparation"
  rm -rf data/lang
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
      "<SIL>" data/local/lang data/lang || exit 1;

  # LM training
  echo "$0: LM training"
  rm -rf data/local/lm/3gram-mincount
  local/train_lms.sh || exit 1;

  # G compilation, check LG composition
  echo "$0: G compilation, check LG composition"
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_test || exit 1;

fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

# mfcc
if [ $stage -le -1 ]; then
  echo "$0: making mfccs"
  for x in train test eval; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $num_jobs data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

# mono
if [ $stage -le 0 ]; then
  echo "$0: train mono model"
  # Make some small data subsets for early system-build stages.
  echo "$0: make training subsets"
  utils/subset_data_dir.sh --shortest data/train 3000 data/train_mono

  # train mono
  steps/train_mono.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $num_jobs \
    data/train_mono data/lang exp/mono || exit 1;

  # Get alignments from monophone system.
  steps/align_si.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/mono exp/mono_ali || exit 1;

  # Monophone decoding
  (
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/mono/graph data/test exp/mono/decode_test
  )&
fi

# tri1
if [ $stage -le 1 ]; then
  echo "$0: train tri1 model"
  # train tri1 [first triphone pass]
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

  # align tri1
  steps/align_si.sh --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

  # decode tri1
  (
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri1/graph data/test exp/tri1/decode_test
  )&
fi

# tri2
if [ $stage -le 2 ]; then
  echo "$0: train tri2 model"
  # train tri2 [delta+delta-deltas]
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

  # align tri2b
  steps/align_si.sh --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

  # decode tri2
  (
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri2/graph data/test exp/tri2/decode_test
  )&
fi

# tri3a
if [ $stage -le 3 ]; then
  echo "$-: train tri3 model"
  # Train tri3a, which is LDA+MLLT,
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;

  # decode tri3a
  (
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri3a/graph data/test exp/tri3a/decode_test
  )&
fi

# tri4
if [ $stage -le 4 ]; then
  echo "$0: train tri4 model"
  # From now, we start building a more serious system (with SAT), and we'll
  # do the alignment with fMLLR.
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" \
    2500 20000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;

  # align tri4a
  steps/align_fmllr.sh  --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri4a exp/tri4a_ali

  # decode tri4a
  (
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri4a/graph data/test exp/tri4a/decode_test
  )&
fi

# tri5
if [ $stage -le 5 ]; then
  echo "$0: train tri5 model"
  # Building a larger SAT system.
  steps/train_sat.sh --cmd "$train_cmd" \
    3500 100000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

  # align tri5a
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;

  # decode tri5
  (
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
     exp/tri5a/graph data/test exp/tri5a/decode_test || exit 1;
  )&
fi

# nnet3 tdnn models
# commented out by default, since the chain model is usually faster and better
#if [ $stage -le 6 ]; then
  # echo "$0: train nnet3 model"
  # local/nnet3/run_tdnn.sh
#fi

# chain model
if [ $stage -le 7 ]; then
  # The iVector-extraction and feature-dumping parts coulb be skipped by setting "--train_stage 7"
  echo "$0: train chain model"
  local/chain/run_tdnn.sh
fi

# getting results (see RESULTS file)
if [ $stage -le 8 ]; then
  echo "$0: extract the results"
  for test_set in test eval; do
  echo "WER: $test_set"
  for x in exp/*/decode_${test_set}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  for x in exp/*/*/decode_${test_set}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  echo

  echo "CER: $test_set"
  for x in exp/*/decode_${test_set}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
  for x in exp/*/*/decode_${test_set}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
  echo
  done
fi

# finish
echo "$0: all done"

exit 0;
