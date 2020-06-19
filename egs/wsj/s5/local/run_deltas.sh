#!/usr/bin/env bash

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script, optionally called from ../run.sh, runs some parts of the recipe
# that used to be present in ../run.sh but which we are now turning into an
# optional extra-- namely, building a delta+delta-delta system.

# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 \
  data/train_si84 data/lang_nosp exp/tri1_ali_si84 exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri2a exp/tri2a/graph_nosp_tgpr || exit 1;

steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri2a/graph_nosp_tgpr \
  data/test_dev93 exp/tri2a/decode_nosp_tgpr_dev93 || exit 1;
steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri2a/graph_nosp_tgpr \
  data/test_eval92 exp/tri2a/decode_nosp_tgpr_eval92 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_bg_5k exp/tri2a exp/tri2a/graph_nosp_bg5k
steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri2a/graph_nosp_bg5k \
  data/test_eval92 exp/tri2a/decode_nosp_eval92_bg5k || exit 1;

steps/decode_fromlats.sh --cmd "$decode_cmd" \
  data/test_dev93 data/lang_nosp_test_tgpr exp/tri2b/decode_nosp_tgpr_dev93 \
  exp/tri2a/decode_nosp_tgpr_dev93_fromlats || exit 1


