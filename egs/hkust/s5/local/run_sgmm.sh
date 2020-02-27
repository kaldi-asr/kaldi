#!/usr/bin/env bash
# Copyright 2016 Alibaba Robotics Corp. (Author: Xingyu Na)
# Apache2.0

# This runs SGMM training.

. ./cmd.sh
. ./path.sh

steps/train_ubm.sh --cmd "$train_cmd" \
  900 data/train data/lang exp/tri5a_ali exp/ubm5a || exit 1;

steps/train_sgmm2.sh --cmd "$train_cmd" \
  14000 35000 data/train data/lang exp/tri5a_ali \
  exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;

utils/mkgraph.sh data/lang_test exp/sgmm2_5a exp/sgmm2_5a/graph || exit 1;
steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri5a/decode \
  exp/sgmm2_5a/graph data/dev exp/sgmm2_5a/decode || exit 1;

exit 0;
