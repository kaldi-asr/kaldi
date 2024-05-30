#!/usr/bin/env bash

# Change this location to somewhere where you want to put the data.
data=$HOME/vystadial_cz

# Load training parameters
. ./env_voip_cs.sh

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data

if [ $stage -le 0 ]; then
  local/download_cs_data.sh $data || exit 1;
fi

lm="build3"

if [ $stage -le 1 ]; then
  local/data_split.sh --every_n 1 $data data "$lm" "dev test"

  local/create_LMs.sh data/local data/train/trans.txt \
    data/test/trans.txt data/local/lm "$lm"

  gzip data/local/lm/$lm

  local/prepare_cs_transcription.sh data/local data/local/dict

  local/create_phone_lists.sh data/local/dict

  utils/prepare_lang.sh data/local/dict '_SIL_' data/local/lang data/lang

  utils/format_lm.sh data/lang data/local/lm/$lm.gz data/local/dict/lexicon.txt data/lang_test

  for part in dev test train; do
    mv data/$part/trans.txt data/$part/text
  done
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc

  for part in dev train; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 10000 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
fi

# train a monophone system
if [ $stage -le 3 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train_10kshort data/lang exp/mono
  (
    utils/mkgraph.sh data/lang_test \
      exp/mono exp/mono/graph
    for test in dev; do
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono/graph \
        data/$test exp/mono/decode_$test
    done
  )&

  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali_train
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono_ali_train exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_test \
      exp/tri1 exp/tri1/graph
    for test in dev; do
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri1/graph \
        data/$test exp/tri1/decode_$test
    done
  )&

  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali_train
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang exp/tri1_ali_train exp/tri2b

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_test \
      exp/tri2b exp/tri2b/graph
    for test in dev; do
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri2b/graph \
        data/$test exp/tri2b/decode_$test
    done
  )&

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_train
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali_train exp/tri3b

  # decode using the tri3b model
  (
    utils/mkgraph.sh data/lang_test \
      exp/tri3b exp/tri3b/graph
    for test in dev; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
        exp/tri3b/graph data/$test \
        exp/tri3b/decode_$test
    done
  )&
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict_sp

  utils/prepare_lang.sh data/local/dict_sp "_SIL_" data/local/lang_tmp data/lang_sp

  utils/format_lm.sh data/lang_sp data/local/lm/$lm.gz data/local/dict_sp/lexicon.txt data/lang_sp_test

  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/train data/lang_sp exp/tri3b exp/tri3b_ali_train_sp
fi

if [ $stage -le 8 ]; then
  # Test the tri3b system with the silprobs and pron-probs.

  # decode using the tri3b model
  utils/mkgraph.sh data/lang_sp_test \
    exp/tri3b exp/tri3b/graph_sp

  for test in dev; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
      exp/tri3b/graph_sp data/$test \
      exp/tri3b/decode_sp_$test
  done
fi

# Train a chain model
if [ $stage -le 9 ]; then
  local/chain/run_tdnn.sh --stage 0
fi

# Don't finish until all background decoding jobs are finished.
wait
