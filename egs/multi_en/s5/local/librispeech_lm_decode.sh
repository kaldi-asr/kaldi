#!/usr/bin/env bash

# Copyright 2016  Allen Guo
# Apache 2.0

# This script decodes the librispeech test set using a librispeech LM, which is assumed
# to be prepared already using the librispeech recipe.

. ./cmd.sh
. ./path.sh
set -e

stage=0
lib=../../librispeech/s5
lang=data/lang_libri_tg_small
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
step=tri5

. utils/parse_options.sh

# You do not need to redo this stage when changing the "step" argument
if [ $stage -le 0 ]; then
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    data/lang $lib/data/local/lm/lm_tgsmall.arpa.gz \
    data/local/dict/lexicon.txt $lang
fi

graph_dir=exp/multi_a/$step/graph_libri_tg
if [ $stage -le 1 ]; then
  utils/mkgraph.sh $lang \
    exp/multi_a/$step $graph_dir
fi

if [ $stage -le 2 ]; then
  steps/decode_fmllr.sh --nj 45 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
    data/librispeech/test exp/multi_a/$step/decode_libri_tg_librispeech
fi
