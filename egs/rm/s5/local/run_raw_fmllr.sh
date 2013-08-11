#!/bin/bash

steps/align_raw_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_raw

steps/train_raw_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali_raw exp/tri3c || exit 1;

utils/mkgraph.sh data/lang exp/tri3c exp/tri3c/graph

steps/decode_raw_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph data/test exp/tri3c/decode

steps/decode_raw_fmllr.sh --use-normal-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph data/test exp/tri3c/decode_2fmllr

