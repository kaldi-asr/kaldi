#!/usr/bin/env bash

# Copyright 2015 Sarah Samson Juan
# Apache 2.0

# This script prepares data and train/decode ASR.
# Download the Iban corpus from github. wav files are in data/wav,
# language model in LM/*.arpa.tar.gz and lexicon in lang/dict.

stage=0

# initialization PATH
. ./path.sh  || die "path.sh expected";
# initialization commands
. ./cmd.sh
. ./utils/parse_options.sh

set -e -o pipefail
corpus=./corpus
# download iban to build ASR
if [ ! -f "$corpus/README" ]; then
    #available from github
    mkdir -p ./$corpus/
    [ ! -f ./iban.tar.gz ] &&  wget http://www.openslr.org/resources/24/iban.tar.gz
    tar xzf iban.tar.gz -C $corpus
fi

nj=16
dev_nj=6

if [ $stage -le 1 ]; then
  echo "Preparing data and training language models"
  local/prepare_data.sh $corpus/
  local/prepare_dict.sh $corpus/
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
  local/prepare_lm.sh
fi


if [ $stage -le 2 ]; then
  # Feature extraction
  for x in train dev; do
      steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
  done
fi

if [ $stage -le 3 ]; then
  ### Monophone
  echo "Starting monophone training."
  utils/subset_data_dir.sh data/train 1000 data/train.1k
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train.1k data/lang exp/mono
  echo "Mono training done."

  (
  echo "Decoding the dev set using monophone models."
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --config conf/decode.config --nj $dev_nj --cmd "$decode_cmd" \
    exp/mono/graph data/dev exp/mono/decode_dev
  echo "Monophone decoding done."
  ) &
fi


if [ $stage -le 4 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd"  \
      3200 30000 data/train data/lang exp/mono_ali exp/tri1
  echo "Triphone training done."

  (
  echo "Decoding the dev set using triphone models."
  utils/mkgraph.sh data/lang_test  exp/tri1 exp/tri1/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd"  \
      exp/tri1/graph  data/dev exp/tri1/decode_dev

  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test/ data/lang_big/ data/dev \
      exp/tri1/decode_dev exp/tri1/decode_dev.rescored
  echo "Triphone decoding done."
  ) &
fi

if [ $stage -le 5 ]; then
  ## Triphones + delta delta
  # Training
  echo "Starting (larger) triphone training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true \
       data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_deltas.sh --cmd "$train_cmd"  \
      4200 40000 data/train data/lang exp/tri1_ali exp/tri2a
  echo "Triphone (large) training done."

  (
  echo "Decoding the dev set using triphone(large) models."
  utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri2a/graph data/dev exp/tri2a/decode_dev

  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test/ data/lang_big/ data/dev \
      exp/tri2a/decode_dev exp/tri2a/decode_dev.rescored
  echo "Triphone(large) decoding done."
  ) &
fi

if [ $stage -le 6 ]; then
  ### Triphone + LDA and MLLT
  # Training
  echo "Starting LDA+MLLT training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd"  \
      data/train data/lang exp/tri2a exp/tri2a_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd"  \
    --splice-opts "--left-context=3 --right-context=3" \
    4200 40000 data/train data/lang exp/tri2a_ali exp/tri2b
  echo "LDA+MLLT training done."

  (
  echo "Decoding the dev set using LDA+MLLT models."
  utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri2b/graph data/dev exp/tri2b/decode_dev

  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test/ data/lang_big/ data/dev \
      exp/tri2b/decode_dev exp/tri2b/decode_dev.rescored
  echo "LDA+MLLT decoding done."
  ) &
fi


if [ $stage -le 7 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/train data/lang exp/tri2b_ali exp/tri3b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri3b/graph  data/dev exp/tri3b/decode_dev

  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test/ data/lang_big/ data/dev \
      exp/tri3b/decode_dev exp/tri3b/decode_dev.rescored
  echo "SAT+FMLLR decoding done."
  ) &
fi


if [ $stage -le 8 ]; then
  echo "Starting SGMM training."
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3b exp/tri3b_ali

  steps/train_ubm.sh --cmd "$train_cmd"  \
      600 data/train data/lang exp/tri3b_ali exp/ubm5b2

  steps/train_sgmm2.sh --cmd "$train_cmd"  \
       5200 12000 data/train data/lang exp/tri3b_ali exp/ubm5b2/final.ubm exp/sgmm2_5b2
  echo "SGMM training done."

  (
  echo "Decoding the dev set using SGMM models"
  # Graph compilation
  utils/mkgraph.sh data/lang_test exp/sgmm2_5b2 exp/sgmm2_5b2/graph
  utils/mkgraph.sh data/lang_big/ exp/sgmm2_5b2 exp/sgmm2_5b2/graph_big

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$decode_cmd" \
      --transform-dir exp/tri3b/decode_dev \
      exp/sgmm2_5b2/graph data/dev exp/sgmm2_5b2/decode_dev

  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test/ data/lang_big/ data/dev \
      exp/sgmm2_5b2/decode_dev exp/sgmm2_5b2/decode_dev.rescored

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$decode_cmd" \
      --transform-dir exp/tri3b/decode_dev \
      exp/sgmm2_5b2/graph_big data/dev exp/sgmm2_5b2/decode_dev.big
  echo "SGMM decoding done."
  ) &
fi

wait;
#score
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
