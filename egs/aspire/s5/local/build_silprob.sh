#!/usr/bin/env bash

set -e

. ./cmd.sh
. ./path.sh

steps/get_prons.sh --cmd "$train_cmd" data/train data/lang exp/tri5a

utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict exp/tri5a/pron_counts_nowb.txt \
  exp/tri5a/sil_counts_nowb.txt exp/tri5a/pron_bigram_counts_nowb.txt data/local/dict_pp

utils/prepare_lang.sh data/local/dict_pp "<unk>" data/local/lang_pp data/lang_pp

cp -rT data/lang_pp data/lang_pp_test
cp -f data/lang_test/G.fst data/lang_pp_test

cp -rT data/lang_pp data/lang_pp_test_fg
cp -f data/lang_test_fg/G.carpa data/lang_pp_test_fg

utils/mkgraph.sh data/lang_pp_test exp/tri5a exp/tri5a/graph_pp
