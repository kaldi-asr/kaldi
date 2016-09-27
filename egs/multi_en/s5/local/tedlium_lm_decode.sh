#!/bin/bash

# Copyright 2016  Allen Guo
# Apache 2.0

# This script decodes the tedlium test set using a tedlium LM, which is assumed
# to be prepared already using the tedlium recipe.

. ./cmd.sh
. ./path.sh
set -e

stage=0
lib=../../tedlium/s5
lang=data/lang_tedlium_tg
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
step=tri5

. utils/parse_options.sh

# You do not need to redo this stage when changing the "step" argument
if [ $stage -le 0 ]; then
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    data/lang $lib/db/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz \
    data/local/dict/lexicon.txt $lang
fi

graph_dir=exp/multi_a/$step/graph_tedlium_tg
if [ $stage -le 1 ]; then
  utils/mkgraph.sh $lang \
    exp/multi_a/$step $graph_dir
fi

if [ $stage -le 2 ]; then
  steps/decode_fmllr.sh --nj 11 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
    data/tedlium/test exp/multi_a/$step/decode_tedlium_tg_tedlium
fi
