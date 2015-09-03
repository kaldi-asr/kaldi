#!/bin/bash

. cmd.sh
set -e

#-steps/align_raw_fmllr.sh --nj 30 --cmd "$train_cmd" --use-graphs true \
#-  data/train_nodup data/lang exp/tri3 exp/tri3_ali_raw_nodup

steps/train_raw_sat.sh   --cmd "$train_cmd" \
   11500 200000 data/train_nodup data/lang exp/tri3_ali_raw_nodup exp/tri4b

graph_dir=exp/tri4b/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log \
  utils/mkgraph.sh data/lang_sw1_tg exp/tri4b $graph_dir
steps/decode_raw_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  --config conf/decode.config \
  $graph_dir data/eval2000 exp/tri4b/decode_eval2000_sw1_tg

# Align the _nodup data with this system
steps/align_raw_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4b exp/tri4b_ali_nodup


# We won't be training the SGMM system on top of this.  Our
# only objective was to get the transforms for DNN training.
