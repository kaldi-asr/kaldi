#!/bin/bash

. ./cmd.sh

set -e

has_fisher=$1

# This runs on all the data data/train_nodup; the triphone baseline, tri4 is
# also trained on that set.

if [ ! -f exp/ubm5/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 1400 data/train_nodup data/lang \
    exp/tri4_ali_nodup exp/ubm5 || exit 1;
fi

# steps/train_sgmm2.sh --cmd "$train_cmd" \
steps/train_sgmm2_group.sh --cmd "$train_cmd" \
  18000 60000 data/train_nodup data/lang exp/tri4_ali_nodup \
  exp/ubm5/final.ubm exp/sgmm2_5 || exit 1;

(
  graph_dir=exp/sgmm2_5/graph_sw1_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_sw1_tg exp/sgmm2_5 $graph_dir
  steps/decode_sgmm2.sh --nj 30 \
    --cmd "$decode_cmd" --config conf/decode.config \
    --transform-dir exp/tri4/decode_eval2000_sw1_tg $graph_dir \
    data/eval2000 exp/sgmm2_5/decode_eval2000_sw1_tg
) &

# Now discriminatively train the SGMM system on data/train_nodup data.
steps/align_sgmm2.sh --nj 50 --cmd "$train_cmd" \
  --transform-dir exp/tri4_ali_nodup \
  --use-graphs true --use-gselect true \
  data/train_nodup data/lang exp/sgmm2_5 exp/sgmm2_5_ali_nodup

# Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 50 --sub-split 30 --num-threads 6 \
  --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4_ali_nodup \
  data/train_nodup data/lang exp/sgmm2_5_ali_nodup exp/sgmm2_5_denlats_nodup

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
  --transform-dir exp/tri4_ali_nodup --boost 0.1 \
  data/train_nodup data/lang exp/sgmm2_5_ali_nodup \
  exp/sgmm2_5_denlats_nodup exp/sgmm2_5_mmi_b0.1

for iter in 1 2 3 4; do
  (
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4/decode_eval2000_sw1_tg \
      data/lang_sw1_tg data/eval2000 \
      exp/sgmm2_5/decode_eval2000_sw1_tg \
      exp/sgmm2_5_mmi_b0.1/decode_eval2000_sw1_tg_it$iter
  ) &
done
wait

if $has_fisher; then
  for iter in 1 2 3 4;do
    (
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        exp/sgmm2_5_mmi_b0.1/decode_eval2000_sw1_{tg,fsh_fg}_it$iter
    ) &
  done
  wait
fi
