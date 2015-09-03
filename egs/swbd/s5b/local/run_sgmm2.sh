#!/bin/bash

. cmd.sh

set -e

# This runs on all the data data/train_nodup; the triphone baseline, tri4b is
# also trained on that set.

if [ ! -f exp/ubm5b/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 1400 data/train_nodup data/lang \
    exp/tri4b_ali_nodup exp/ubm5b || exit 1;
fi 

steps/train_sgmm2.sh --cmd "$train_cmd" \
  18000 60000 data/train_nodup data/lang exp/tri4b_ali_nodup \
  exp/ubm5b/final.ubm exp/sgmm2_5b || exit 1;

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/sgmm2_5b/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/sgmm2_5b $graph_dir
    steps/decode_sgmm2.sh --nj 30 \
      --cmd "$decode_cmd" --config conf/decode.config \
      --transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} $graph_dir \
      data/eval2000 exp/sgmm2_5b/decode_eval2000_sw1_${lm_suffix}
  ) &
done

# Now discriminatively train the SGMM system on data/train_nodup data.
steps/align_sgmm2.sh --nj 100 --cmd "$train_cmd" \
  --transform-dir exp/tri4b_ali_nodup \
  --use-graphs true --use-gselect true \
  data/train_nodup data/lang exp/sgmm2_5b exp/sgmm2_5b_ali_nodup

# Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 100 --sub-split 30 --num-threads 6 \
  --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4b_ali_nodup \
  data/train_nodup data/lang exp/sgmm2_5b_ali_nodup exp/sgmm2_5b_denlats_nodup

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
  --transform-dir exp/tri4b_ali_nodup --boost 0.1 \
  data/train_nodup data/lang exp/sgmm2_5b_ali_nodup \
  exp/sgmm2_5b_denlats_nodup exp/sgmm2_5b_mmi_b0.1

for iter in 1 2 3 4; do
  for lm_suffix in tg fsh_tgpr; do
    (
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} \
        data/lang_sw1_${lm_suffix} data/eval2000 \
        exp/sgmm2_5b/decode_eval2000_sw1_${lm_suffix} \
        exp/sgmm2_5b_mmi_b0.1/decode_eval2000_sw1_${lm_suffix}_it$iter
    ) &
  done
done
wait

for iter in 1 2 3 4;do
  (
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_fsh_{tgpr,fg} data/eval2000 \
      exp/sgmm2_5b_mmi_b0.1/decode_eval2000_sw1_fsh_{tgpr,fg}_it$iter
  ) &
done
wait
