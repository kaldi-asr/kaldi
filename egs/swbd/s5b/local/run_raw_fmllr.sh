#!/bin/bash

. cmd.sh
set -e

steps/align_raw_fmllr.sh --nj 30 --cmd "$train_cmd" --use-graphs true \
  data/train_100k_nodup data/lang exp/tri3b exp/tri3b_ali_100k_nodup

steps/train_raw_sat.sh   --cmd "$train_cmd" \
   5500 90000 data/train_100k_nodup data/lang exp/tri3b_ali_100k_nodup \
   exp/tri4d

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4d/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4d $graph_dir
    steps/decode_raw_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data/eval2000 exp/tri4d/decode_eval2000_sw1_${lm_suffix}
    steps/decode_raw_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data/train_dev exp/tri4d/decode_train_dev_sw1_${lm_suffix}
  ) &
done
wait

# align 100k_nodup data with this system
steps/align_raw_fmllr.sh --nj 30 --cmd "$train_cmd" --use-graphs true \
  data/train_100k_nodup data/lang exp/tri4d exp/tri4d_ali_100k_nodup

# also align the _nodup data with this system
steps/align_raw_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4d exp/tri4d_ali_nodup


# We won't be training the SGMM system on top of this.  Our
# only objective was to get the transforms for DNN training.
