#!/bin/bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#                2018  Ke Li

# Begin configuration section.
# initialize path and commands
. ./path.sh
. ./cmd.sh

nj=9 # number of parallel jobs
stage=0

set -e


[ ! -L "steps" ] && ln -s ../../wsj/s5/steps

[ ! -L "utils" ] && ln -s ../../wsj/s5/utils

[ ! -L "conf" ] && ln -s ../../wsj/s5/conf

. utils/parse_options.sh # accept options
AUDIO=(
    /export/corpora5/LDC/LDC2015S04/data/conversation/audio/
    /export/corpora5/LDC/LDC2015S04/data/interview/audio/
)
TEXT=(
    /export/corpora5/LDC/LDC2015S04/data/conversation/transcript/phaseII
    /export/corpora5/LDC/LDC2015S04/data/interview/transcript/phaseII
)

# TODO Data preparation
if [ $stage -le 0 ]; then
  # 01NC01FBX_0101 start_time end_time LID utt_trans
  # 01NC01FBX is spkeaer id
  # 0101 part 01
  local/prepare_text_data.sh $TEXT
  local/prepare_audio_data.sh $AUDIO
fi

if [ $stage -le 1 ]; then
  #TODO Prepare dictionary
  #local/prepare_dict.sh $corpus
  utils/validate_dict_dir.pl data/local/dict_nosp
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
  utils/validate_lang.pl data/lang_nosp
fi

# Prepare language models
if [ $stage -le 2 ]; then
  local/train_lms_srilm.sh --oov-symbol "<unk>" --words-file \
    data/lang_nosp/words.txt data data/lm
  utils/format_lm.sh data/lang_nosp data/lm/lm.gz \
    data/local/dict_nosp/lexiconp.txt data/lang_nosp_test
  utils/validate_lang.pl data/lang_nosp_test
fi

# Feature extraction
if [ $stage -le 3 ]; then
  for x in train dev_man dev_sge; do
    dir=data/$x
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $dir
    utils/fix_data_dir.sh $dir
    steps/compute_cmvn_stats.sh $dir

    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

if [ $stage -le 4 ]; then
  # Make some small data subsets for early system-build stages.
  # For the monophone stages we select the shortest utterances, which should make it easier to align the data from a flat start.
  utils/subset_data_dir.sh --shortest data/train 5000 data/train_short
fi

if [ $stage -le 5 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_short data/lang_nosp_test exp/mono
  # decode using the monophone model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/mono exp/mono/graph_nosp
    for test in dev_man dev_sge; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/mono/graph_nosp \
      data/$test exp/mono/decode_nosp_$test
    done
  )&
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali || exit 1;
  # train a first delta + delta-delta triphone system on all utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train data/lang_nosp_test exp/mono_ali exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri1 exp/tri1/graph_nosp
    for test in dev_man dev_sge; do
      nspk=$(wc -l <data/${test}/spk2utt)
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" exp/tri1/graph_nosp \
      data/$test exp/tri1/decode_nosp_$test
    done
  )&
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri1 exp/tri1_ali || exit 1;
  
  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang_nosp_test exp/tri1_ali exp/tri2

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri2 exp/tri2/graph_nosp
    for test in dev_man dev_sge; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri2/graph_nosp \
      data/$test exp/tri2/decode_nosp_$test
    done
  )&
fi

if [ $stage -le 8 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/train data/lang_nosp_test exp/tri2 exp/tri2_ali || exit 1;

  # train tri3, which is a LDA+MLLT+SAT system on all utterances
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang_nosp_test exp/tri2_ali exp/tri3

  # decode using the tri3 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph_nosp
    for test in dev_man dev_sge; do
      steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" exp/tri3/graph_nosp \
      data/$test exp/tri3/decode_nosp_$test
    done
  )&
fi

if [ $stage -le 9 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt data/local/dict
  
  utils/prepare_lang.sh data/local/dict \
    "<unk>" data/local/lang data/lang

  utils/format_lm.sh data/lang data/lm/lm.gz \
    data/local/dict/lexiconp.txt data/lang_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_test exp/tri3 exp/tri3_ali

  # decode using the tri3 model with pronunciation and silence probabilities
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph
    for test in dev_man dev_sge; do
      steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" exp/tri3/graph \
      data/$test exp/tri3/decode_$test
    done
  )&
fi

# After run.sh is finished, run the followings:
# ./local/chain/tuning/run_tdnn_1a.sh
# ./local/rnnlm/run_tdnn_lstm_1a.sh
exit 0;
