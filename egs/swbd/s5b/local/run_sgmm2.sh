#!/bin/bash

. cmd.sh

# This runs on just the 100k_nodup data; the triphone baseline, tri4a, is
# also trained on that subset.


if [ ! -f exp/ubm5a/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 700 data/train_100k_nodup data/lang \
    exp/tri4a_ali_100k_nodup exp/ubm5a || exit 1;
fi 

steps/train_sgmm2.sh --cmd "$train_cmd" \
  9000 30000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup \
  exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/sgmm2_5a/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/sgmm2_5a $graph_dir
    steps/decode_sgmm2.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} $graph_dir \
      data/eval2000 exp/sgmm2_5a/decode_eval2000_sw1_${lm_suffix}
  ) &
done

 # Now discriminatively train the SGMM system on 100k_nodup data.
steps/align_sgmm2.sh --nj 50 --cmd "$train_cmd" --transform-dir exp/tri4a_ali_100k_nodup \
  --use-graphs true --use-gselect true data/train_100k_nodup data/lang exp/sgmm2_5a exp/sgmm2_5a_ali_100k_nodup

  # Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 50 --sub-split 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4a_ali_100k_nodup \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup exp/sgmm2_5a_mmi_b0.1

for iter in 1 2 3 4; do
  for lm_suffix in tg fsh_tgpr; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
     data/lang_sw1_${lm_suffix} data/eval2000 \
     exp/sgmm2_5a/decode_eval2000_sw1_${lm_suffix} \
     exp/sgmm2_5a_mmi_b0.1/decode_eval2000_sw1_${lm_suffix}_it$iter 
  done
done




 
