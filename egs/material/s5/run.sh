#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
. ./path.sh
. ./cmd.sh

nj=30 # number of parallel jobs
stage=1
. utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

corpus=/export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/
#corpus=/export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1B-BUILD_v1.0/

if [ $stage -le 1 ]; then
  local/prepare_text_data.sh $corpus
  local/prepare_audio_data.sh $corpus
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $corpus
  utils/validate_dict_dir.pl data/local/dict_nosp
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
  utils/validate_lang.pl data/lang_nosp
fi

if [ $stage -le 3 ]; then
  local/train_lms_srilm.sh --oov-symbol "<unk>" --words-file \
    data/lang_nosp/words.txt data data/lm
  utils/format_lm.sh data/lang_nosp data/lm/lm.gz \
    data/local/dict_nosp/lexiconp.txt data/lang_nosp_test
  utils/validate_lang.pl data/lang_nosp_test
fi

if [ $stage -le 4 ]; then
  for set in train dev; do
    dir=data/$set
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

# Create a subset with 40k short segments to make flat-start training easier
if [ $stage -le 5 ]; then
  utils/subset_data_dir.sh --shortest data/train 40000 data/train_40kshort
fi

# monophone training
if [ $stage -le 6 ]; then
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_40kshort data/lang_nosp_test exp/mono
  (
    utils/mkgraph.sh data/lang_nosp_test \
      exp/mono exp/mono/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/mono/graph_nosp \
        data/$test exp/mono/decode_nosp_$test
    done
  )&

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/mono exp/mono_ali
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 7 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    2000 30000 data/train data/lang_nosp_test exp/mono_ali exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri1 exp/tri1/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri1/graph_nosp \
        data/$test exp/tri1/decode_nosp_$test
    done
  )&

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri1 exp/tri1_ali
fi

# train an LDA+MLLT system.
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 3000 60000 \
    data/train data/lang_nosp_test exp/tri1_ali exp/tri2

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri2 exp/tri2/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri2/graph_nosp \
        data/$test exp/tri2/decode_nosp_$test
    done
  )&

  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/train data/lang_nosp_test exp/tri2 exp/tri2_ali
fi

# Train tri3, which is LDA+MLLT+SAT
if [ $stage -le 9 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 6000 80000 \
    data/train data/lang_nosp_test exp/tri2_ali exp/tri3

  # decode using the tri3 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph_nosp
    for test in dev; do
      steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" exp/tri3/graph_nosp \
        data/$test exp/tri3/decode_nosp_$test
    done
  )&
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 10 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp_test exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

  utils/format_lm.sh data/lang data/lm/lm.gz \
    data/local/dict/lexiconp.txt data/lang_test

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_test exp/tri3 exp/tri3_ali
fi

if [ $stage -le 11 ]; then
  # Test the tri3 system with the silprobs and pron-probs.

  # decode using the tri3 model
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph
  for test in dev; do
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
      exp/tri3/graph data/$test exp/tri3/decode_$test
  done
fi
