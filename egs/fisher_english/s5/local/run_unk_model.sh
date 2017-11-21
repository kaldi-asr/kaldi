#!/bin/bash

# Copyright 2017  Vimal Manohar

utils/lang/make_unk_lm.sh data/local/dict exp/unk_lang_model || exit 1

utils/prepare_lang.sh --unk-fst exp/unk_lang_model/unk_fst.txt \
  data/local/dict "<unk>" data/local/lang data/lang_unk

# note: it's important that the LM we built in data/lang/G.fst was created using
# pocolm with the option --limit-unk-history=true (see ted_train_lm.sh).  This
# keeps the graph compact after adding the unk model (we only have to add one
# copy of it).

mkdir -p data/lang_poco_test_unk
cp -r data/lang_unk/* data/lang_poco_test_unk
cp data/lang_poco_test/G.fst data/lang_poco_test_unk/G.fst

mkdir -p data/lang_poco_test_ex250k_unk
cp -r data/lang_unk/* data/lang_poco_test_ex250k_unk
cp data/lang_poco_test_ex250k/G.fst data/lang_poco_test_ex250k_unk/G.fst

exit 0

# utils/mkgraph.sh data/lang_unk exp/tri3 exp/tri3/graph_unk
